"""ASFormer 风格全序列分割模型（局部膨胀卷积 + 全局自注意力 + 多阶段精化解码器）。

**与既有 ``transformer`` 的关系**：``transformer`` 是纯 Encoder + 逐帧线性头，在本项目
（9.5k 帧、144 维手工特征）上 edit 比 MS-TCN++ 低 24.24（见
``docs/experiments/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md`` §5）。ASFormer（Ding et al. 2021）
的关键差异有两处，正是本实现要检验的假设：

1. **每个 block 先做局部膨胀卷积再做全局注意力**——注意力之前先注入时序局部性先验，
   避免纯注意力在**小数据**上直接过拟合到逐帧线索（本数据集 train AUC 可到 0.999、
   test AUC 掉到 0.46，见 ``tools/probe_direction.py``）。
2. **多阶段精化解码器**：解码器逐级以前一级的 softmax 概率为输入，并用**交叉注意力**
   回看编码器输出；每级都算损失（深监督）。这与 MS-TCN++ 的多 stage 精化同源，
   但级间用注意力而非纯膨胀卷积。

归一化沿用本仓库"checkpoint 自描述"不变量：z-score 统计以持久 buffer 内置，随
``state_dict`` 存取，训练前由流水线调用 ``fit_normalization`` 按训练集写入。

**非因果**（注意力看整段序列）→ 只能进 ``full_sequence_temporal`` 流水线；滑窗流水线会因
``causal=False`` 拒绝。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mstcn import _apply_sequence_normalization, _resolve_sequence_normalization


def _sinusoidal_position(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """正弦位置编码 ``[T, dim]``（不引入可学习参数，短序列上比可学习编码更稳）。"""

    position = torch.arange(length, device=device).float().unsqueeze(1)
    index = torch.arange(dim, device=device).float().unsqueeze(0)
    divisor = torch.exp(torch.floor(index / 2) * (-np.log(10000.0) / max(dim, 1)))
    encoded = position * divisor
    output = torch.zeros(length, dim, device=device)
    output[:, 0::2] = torch.sin(encoded[:, 0::2])
    output[:, 1::2] = torch.cos(encoded[:, 1::2])
    return output


class EncoderBlock(nn.Module):
    """编码器 block：局部膨胀 conv → 多头自注意力 → FFN，三处前置 LayerNorm 残差。

    ``x`` 形状 ``[B, T, H]``；第 ``i`` 层的膨胀率取 ``2^(i % 4)``，兼顾细/粗时间尺度。
    """

    def __init__(self, hidden: int, heads: int, dilation: int, dropout: float):
        super().__init__()
        self.local_norm = nn.LayerNorm(hidden)
        self.local = nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation)
        self.attn_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(self.local_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.dropout(F.relu(local))
        attention, _ = self.attn(*([self.attn_norm(x)] * 3), need_weights=False)
        x = x + self.dropout(attention)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class DecoderLayer(nn.Module):
    """精化解码层：自身自注意力 + 对编码器输出的交叉注意力 + FFN。

    第 1 层输入是编码器输出（``hidden`` 维，``project_input=False`` 直通）；其后每层输入是
    **上一层的 softmax 概率**（``num_classes`` 维，``project_input=True`` 经 1×1 投影回
    ``hidden`` 维）——这是 ASFormer 的级联精化机制。

    ``project_input`` 显式声明而非按张量末维推断：当 ``num_classes == hidden`` 时形状判断会
    静默跳过投影，属于"配了却不生效"的隐式分支，显式开关避免该类静默错误。
    """

    def __init__(self, hidden: int, heads: int, in_dim: int, dropout: float, project_input: bool):
        super().__init__()
        self.project_input = project_input
        self.in_proj = (
            nn.Conv1d(in_dim, hidden, kernel_size=1) if project_input else None
        )
        self.self_norm = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x.transpose(1, 2)).transpose(1, 2) if self.project_input else x
        attention, _ = self.self_attn(*([self.self_norm(z)] * 3), need_weights=False)
        z = z + self.dropout(attention)
        cross, _ = self.cross_attn(self.cross_norm(z), memory, memory, need_weights=False)
        z = z + self.dropout(cross)
        return z + self.dropout(self.ffn(self.ffn_norm(z)))


class ASFormerNet(nn.Module):
    """ASFormer 风格分割网络：编码器 Stack + 多阶段精化解码器（深监督）。

    ``forward`` 输入 ``[B, T, F]``、输出 ``[B, T, C]``（**仅最后一级解码器**，供推理/评估）；
    训练损失经 ``compute_loss`` 汇总所有解码级（加权 CE + 可选 T-MSE）。

    ``num_decoders`` 为精化级数（≥1）；``1`` 时退化为"编码器 + 单头"，是本模型的容量下界。
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 128,
        heads: int = 4,
        num_encoders: int = 5,
        num_decoders: int = 3,
        dropout: float = 0.3,
        tmse_weight: float = 0.15,
        tmse_clip: float = 4.0,
        sequence_normalization: str = "none",
    ):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"ASFormer hidden 必须能被 heads 整除: hidden={hidden}, heads={heads}")
        self.sequence_normalization = _resolve_sequence_normalization(sequence_normalization)
        self.num_classes = num_classes
        self.tmse_weight = tmse_weight
        self.tmse_clip = tmse_clip
        self.input_projection = nn.Linear(input_dim, hidden)
        self.encoders = nn.ModuleList(
            EncoderBlock(hidden, heads, dilation=2 ** (index % 4), dropout=dropout)
            for index in range(num_encoders)
        )
        # 第 1 级吃编码器输出（hidden 维，直通），其后各级吃上一级概率（num_classes 维，需投影）。
        self.decoders = nn.ModuleList(
            DecoderLayer(
                hidden, heads,
                in_dim=hidden if index == 0 else num_classes,
                dropout=dropout,
                project_input=index > 0,
            )
            for index in range(num_decoders)
        )
        self.classifier = nn.Conv1d(hidden, num_classes, kernel_size=1)
        # 归一化统计随 state_dict 持久化：[1, 1, F]，初值直通。
        self.register_buffer("norm_mean", torch.zeros(1, 1, input_dim))
        self.register_buffer("norm_std", torch.ones(1, 1, input_dim))

    def _forward_stages(self, x: torch.Tensor) -> list[torch.Tensor]:
        """返回各级解码器 logits 列表，每个 ``[B, C, T]``。"""

        x = (x - self.norm_mean) / self.norm_std
        x = _apply_sequence_normalization(x, self.sequence_normalization)
        z = self.input_projection(x)
        z = z + _sinusoidal_position(z.shape[1], z.shape[2], z.device).unsqueeze(0)
        for encoder in self.encoders:
            z = encoder(z)
        memory = z
        outputs: list[torch.Tensor] = []
        stage_input: torch.Tensor = z
        for decoder in self.decoders:
            z = decoder(stage_input, memory)
            logits = self.classifier(z.transpose(1, 2))  # [B, C, T]
            outputs.append(logits)
            stage_input = F.softmax(logits, dim=1).transpose(1, 2)  # 下一级吃概率
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 推理约定：只取最后一级解码器，转回 [B, T, C]。
        return self._forward_stages(x)[-1].transpose(1, 2)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """训练配方：每级解码器的加权 CE + λ·T-MSE（口径同 ``mstcn2``，便于架构对照）。"""

        total = x.new_zeros(())
        for logits in self._forward_stages(x):
            ce = criterion(logits.transpose(1, 2).reshape(-1, self.num_classes), y.reshape(-1))
            log_p = F.log_softmax(logits, dim=1)
            tmse = torch.clamp(
                (log_p[:, :, 1:] - log_p[:, :, :-1].detach()) ** 2, min=0.0, max=self.tmse_clip ** 2
            ).mean()
            total = total + ce + self.tmse_weight * tmse
        return total

    def normalizer_spec(self) -> str:
        """归一化口径的溯源声明（本模型恒按训练集 z-score 归一化）。"""

        return ("zscore/train-set/buffers/v1" if self.sequence_normalization == "none"
                else f"zscore/train-set/buffers/v1+sequence-{self.sequence_normalization}/v1")

    def fit_normalization(self, features: list) -> None:
        """训练前钩子：按训练集 z-score 统计写入归一化 buffer（口径同 ``mstcn`` / ``mstcn2``）。"""

        arr = np.concatenate(features, axis=0)  # [ΣT, F]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std[std < 1e-4] = 1.0
        dev = self.norm_mean.device
        self.norm_mean.copy_(torch.tensor(mean, dtype=torch.float32, device=dev).reshape(1, 1, -1))
        self.norm_std.copy_(torch.tensor(std, dtype=torch.float32, device=dev).reshape(1, 1, -1))
