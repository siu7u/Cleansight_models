"""MS-TCN++ 离线分割模型（多 stage 深监督 + T-MSE，纯 nn.Module + 自带输入归一化）。

与简化版 ``mstcn`` 的区别，也正是"++"的两处要点：

1. **多 stage 精化（deep supervision）**：一个预测生成 stage 之后串接若干精化 stage，后一
   stage 以前一 stage 的 softmax 概率为输入、逐步修正过分割。训练时**每个 stage 都算损失**
   （深监督），推理只取最后一个 stage。
2. **双膨胀预测生成层（dual dilated layer, Li et al. 2020）**：预测生成 stage 的每层并联两
   条膨胀卷积（正序 ``2^i`` 与逆序 ``2^(L-1-i)``），兼顾大小时间尺度——这是 MS-TCN2 相对
   MS-TCN 的结构改动。精化 stage 仍用单膨胀残差层。

**T-MSE 平滑损失**：对每个 stage 的逐帧 log-softmax，惩罚相邻帧的突变（截断的均方差），
抑制离线全序列常见的碎段抖动。总损失 = Σ_stage (加权 CE + λ·T-MSE)。CE 的类别权重由流水
线拥有（随数据分布走），以 ``criterion`` 传入；T-MSE 的 λ/τ 是模型自身配方，写在这里。

监督配方（多 stage + T-MSE）随架构走，故经 duck-type 钩子 ``compute_loss`` 暴露给全序列
流水线（与 ``fit_normalization`` 同一模式）；``forward`` 仍守 ``[B,T,F]->[B,T,C]`` 推理约定
（只出最后一个 stage）。**双向（非因果）**，只适用于全序列离线流水线，滑窗流水线会因
``causal=False`` 拒绝它。

输入归一化（按训练集 z-score）以注册 buffer 内置，随 state_dict 存取、train/eval 自动一致
应用——同 ``mstcn``，符合"checkpoint 自描述"不变量。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):
    """单膨胀残差层：膨胀 conv3 → ReLU → 1×1 conv → Dropout，残差相加（精化 stage 用）。"""

    def __init__(self, dilation: int, channels: int, dropout: float):
        super().__init__()
        self.conv_dilated = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        return x + self.dropout(out)


class DualDilatedResidualLayer(nn.Module):
    """双膨胀残差层（MS-TCN2 预测生成 stage 用）：并联正序/逆序两条膨胀卷积再融合。

    第 ``i`` 层（共 ``num_layers`` 层）并联膨胀率 ``2^i``（细）与 ``2^(num_layers-1-i)``（粗）
    两条 conv3，拼接后经 1×1 融合，兼顾大小时间尺度。
    """

    def __init__(self, i: int, num_layers: int, channels: int, dropout: float):
        super().__init__()
        d1, d2 = 2 ** i, 2 ** (num_layers - 1 - i)
        self.conv_fine = nn.Conv1d(channels, channels, kernel_size=3, padding=d1, dilation=d1)
        self.conv_coarse = nn.Conv1d(channels, channels, kernel_size=3, padding=d2, dilation=d2)
        self.conv_1x1 = nn.Conv1d(2 * channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(torch.cat([self.conv_fine(x), self.conv_coarse(x)], dim=1))
        out = self.conv_1x1(out)
        return x + self.dropout(out)


class PredictionGeneration(nn.Module):
    """预测生成 stage：1×1 投影 → 若干双膨胀层 → 1×1 分类头。输入/输出 [B, C, T]。"""

    def __init__(self, in_dim: int, num_classes: int, channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, channels, kernel_size=1)
        self.layers = nn.ModuleList(
            DualDilatedResidualLayer(i, num_layers, channels, dropout) for i in range(num_layers)
        )
        self.out = nn.Conv1d(channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        for layer in self.layers:
            z = layer(z)
        return self.out(z)


class Refinement(nn.Module):
    """精化 stage：以前一 stage 概率为输入，1×1 投影 → 单膨胀层(2^i) → 1×1 分类头。"""

    def __init__(self, num_classes: int, channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Conv1d(num_classes, channels, kernel_size=1)
        self.layers = nn.ModuleList(
            DilatedResidualLayer(2 ** i, channels, dropout) for i in range(num_layers)
        )
        self.out = nn.Conv1d(channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        for layer in self.layers:
            z = layer(z)
        return self.out(z)


class MSTCN2(nn.Module):
    """MS-TCN++：预测生成 stage + 若干精化 stage，深监督 + T-MSE。

    ``forward`` 输入 ``[B, T, F]``、输出 ``[B, T, C]``（**仅最后一个 stage**，供推理/评估）；
    训练损失经 ``compute_loss`` 汇总所有 stage。归一化 buffer 初值 0/1（直通），由
    ``fit_normalization`` 在训练前按训练集统计写入。
    """

    def __init__(
        self,
        in_dim: int,
        classes: int,
        hidden: int = 64,
        num_stages: int = 4,
        num_layers: int = 10,
        dropout: float = 0.3,
        tmse_weight: float = 0.15,
        tmse_clip: float = 4.0,
    ):
        super().__init__()
        self.num_classes = classes
        self.tmse_weight = tmse_weight
        self.tmse_clip = tmse_clip
        self.stage0 = PredictionGeneration(in_dim, classes, hidden, num_layers, dropout)
        self.refines = nn.ModuleList(
            Refinement(classes, hidden, num_layers, dropout) for _ in range(num_stages - 1)
        )
        # 归一化统计随 state_dict 持久化：[1, 1, F]，初值直通。
        self.register_buffer("norm_mean", torch.zeros(1, 1, in_dim))
        self.register_buffer("norm_std", torch.ones(1, 1, in_dim))

    def _forward_stages(self, x: torch.Tensor) -> list[torch.Tensor]:
        """返回各 stage 的 logits 列表，每个 ``[B, C, T]``（内部 Conv1d 布局）。"""
        x = (x - self.norm_mean) / self.norm_std
        z = x.transpose(1, 2)  # [B, F, T]
        out = self.stage0(z)
        outputs = [out]
        for refine in self.refines:
            out = refine(F.softmax(out, dim=1))
            outputs.append(out)
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 推理约定：只取最后一个 stage，转回 [B, T, C]。
        return self._forward_stages(x)[-1].transpose(1, 2)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """训练配方（duck-type 钩子）：每个 stage 的加权 CE + λ·T-MSE 求和。

        ``criterion`` 为流水线拥有的类别加权 CE（监督口径随数据走）；T-MSE 是模型自身的
        平滑先验，惩罚相邻帧 log-softmax 的突变（截断均方差，上界 τ²）。
        """
        total = x.new_zeros(())
        for logits in self._forward_stages(x):  # [B, C, T]
            ce = criterion(logits.transpose(1, 2).reshape(-1, self.num_classes), y.reshape(-1))
            log_p = F.log_softmax(logits, dim=1)
            # 相邻帧 log-prob 的截断均方差；前一帧 detach（同官方实现，避免双向反传放大）。
            tmse = torch.clamp(
                (log_p[:, :, 1:] - log_p[:, :, :-1].detach()) ** 2, min=0.0, max=self.tmse_clip ** 2
            ).mean()
            total = total + ce + self.tmse_weight * tmse
        return total

    def fit_normalization(self, features: list) -> None:
        """训练前可选钩子：按训练集 z-score 统计写入归一化 buffer（同 ``mstcn``）。"""
        arr = np.concatenate(features, axis=0)  # [ΣT, F]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std[std < 1e-4] = 1.0
        dev = self.norm_mean.device
        self.norm_mean.copy_(torch.tensor(mean, dtype=torch.float32, device=dev).reshape(1, 1, -1))
        self.norm_std.copy_(torch.tensor(std, dtype=torch.float32, device=dev).reshape(1, 1, -1))
