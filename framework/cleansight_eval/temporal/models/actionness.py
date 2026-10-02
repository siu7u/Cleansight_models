"""动作性解耦的双分支分割模型（ASRF 思路）：把"有没有动作"与"是哪个动作"拆成两个头。

**立项依据（本项目实测）**：当前最佳配方（`mstcn2` s4l10h128 + `roi-grid-144`，8 seed）
在 test 上的主要误差**不是段边界，而是整段被吞成 `idle`**——逐帧召回
`short_brush_cleaning` 0.0%、`flush` 9.1%、`long_brush_withdraw` 5.8%、
`long_brush_insert` 30.2%；而且模型的帧准确率 53.98 **低于平凡基线"全预测 idle"的 55.25**。
复现见 ``docs/experiments/EXPERIMENT_REPORT_ARCH_SURVEY_20260927.md`` §1。

单头 softmax 下，"是否存在动作"与"是哪个动作"共享同一组 logits：`idle` 因为压倒性
多数而不断吸走概率质量，稀类动作整段被压掉。本模型把输出**重参数化**为

    P(idle | t)  = 1 − a(t)
    P(c    | t)  = a(t) · softmax(class_logits)[c],   c ≠ idle

其中 ``a(t)`` 是**类别无关**的 actionness 概率（单标量）。于是"这段到底有没有动作"只由
一个标量决定、并被所有非 idle 类的梯度共同推高，而不是被 6 路 softmax 内部竞争稀释。

**与 ``mstcn2`` 的关系**：主干结构完全一致（复用 ``mstcn2`` 的双膨胀预测生成层与单膨胀
精化层、同样的多 stage 深监督与 T-MSE），**只改输出参数化与损失组合**。因此本模型与
``mstcn2`` 的对照是干净的"输出参数化"单变量对照，不混入容量或特征差异。

组合出的 ``log p`` 是**已归一化的对数概率**：把它直接喂给流水线的类别加权
``CrossEntropyLoss`` 时，内部 ``log_softmax`` 对已归一化向量是恒等映射，因此监督口径
与其它模型完全一致，流水线无需任何改动。

**非因果** → 只能进 ``full_sequence_temporal`` 流水线。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mstcn import _apply_sequence_normalization, _resolve_sequence_normalization
from .mstcn2 import DilatedResidualLayer, DualDilatedResidualLayer


class _StageTrunk(nn.Module):
    """一个 stage 的主干特征提取（不接分类头）：1×1 投影 → 若干残差层 → ``[B, H, T]``。

    ``dual=True`` 用双膨胀层（预测生成 stage），``dual=False`` 用单膨胀层（精化 stage）。
    """

    def __init__(self, in_dim: int, channels: int, num_layers: int, dropout: float, dual: bool):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, channels, kernel_size=1)
        if dual:
            self.layers = nn.ModuleList(
                DualDilatedResidualLayer(i, num_layers, channels, dropout) for i in range(num_layers)
            )
        else:
            self.layers = nn.ModuleList(
                DilatedResidualLayer(2 ** i, channels, dropout) for i in range(num_layers)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        for layer in self.layers:
            z = layer(z)
        return z


class _DualHead(nn.Module):
    """一个 stage 的双头：actionness 单标量 + 非 idle 类条件分布。"""

    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.actionness = nn.Conv1d(channels, 1, kernel_size=1)
        self.action_class = nn.Conv1d(channels, num_classes - 1, kernel_size=1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ``(actionness_logit [B,1,T], class_logits [B,C-1,T])``。"""

        return self.actionness(z), self.action_class(z)


def compose_log_probability(
    actionness_logit: torch.Tensor, class_logits: torch.Tensor
) -> torch.Tensor:
    """把 actionness 与类别条件分布组合成 ``[B, C, T]`` 的**对数概率**。

    写法用 ``logsigmoid`` 保证数值稳定（避免 ``log(1−sigmoid(x))`` 在 x 很大时下溢）：

        log P(idle) = logsigmoid(−x)
        log P(c)    = logsigmoid(x) + log_softmax(class_logits)[c]

    返回的向量在概率空间和为 1，故可**直接**作为 logits 传入 ``CrossEntropyLoss``
    （``log_softmax`` 对已归一化的对数概率是恒等映射）。
    """

    class_log_p = F.log_softmax(class_logits, dim=1)
    idle_log_p = F.logsigmoid(-actionness_logit)
    action_log_p = F.logsigmoid(actionness_logit) + class_log_p
    return torch.cat([idle_log_p, action_log_p], dim=1)


class ActionnessMSTCN2(nn.Module):
    """MS-TCN++ 主干 + actionness/类别双分支输出（多 stage 深监督 + T-MSE）。

    ``forward`` 输入 ``[B, T, F]``、输出 ``[B, T, C]`` 的**对数概率**（已归一化，等价于
    logits 用法）；训练损失经 ``compute_loss`` 汇总所有 stage。

    ``actionness_aux_weight``：类别无关 actionness 的附加 BCE 权重。
    ``0.0``（默认）表示**纯输出重参数化**——actionness 只通过组合后的 CE 拿梯度，
    用于隔离"输出参数化"这一单变量的效应；``>0`` 则额外用 ``y != idle`` 直接监督
    actionness 分支（ASRF 口径的显式动作性监督）。
    """

    def __init__(
        self,
        in_dim: int,
        classes: int,
        hidden: int = 128,
        num_stages: int = 4,
        num_layers: int = 10,
        dropout: float = 0.3,
        tmse_weight: float = 0.15,
        tmse_clip: float = 4.0,
        actionness_aux_weight: float = 0.0,
        sequence_normalization: str = "none",
    ):
        super().__init__()
        if classes < 2:
            raise ValueError(f"num_classes 至少为 2（需含 idle 与至少一个动作类），实际 {classes}")
        self.sequence_normalization = _resolve_sequence_normalization(sequence_normalization)
        self.num_classes = classes
        self.tmse_weight = tmse_weight
        self.tmse_clip = tmse_clip
        self.actionness_aux_weight = float(actionness_aux_weight)
        self.stage0 = _StageTrunk(in_dim, hidden, num_layers, dropout, dual=True)
        self.refine_trunks = nn.ModuleList(
            _StageTrunk(classes, hidden, num_layers, dropout, dual=False) for _ in range(num_stages - 1)
        )
        self.heads = nn.ModuleList(_DualHead(hidden, classes) for _ in range(num_stages))
        # 归一化统计随 state_dict 持久化：[1, 1, F]，初值直通。
        self.register_buffer("norm_mean", torch.zeros(1, 1, in_dim))
        self.register_buffer("norm_std", torch.ones(1, 1, in_dim))

    def _forward_stages(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """返回各 stage 的 ``(actionness_logit, class_logits)`` 列表。"""

        x = (x - self.norm_mean) / self.norm_std
        x = _apply_sequence_normalization(x, self.sequence_normalization)
        z = self.stage0(x.transpose(1, 2))  # [B, H, T]
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, head in enumerate(self.heads):
            if index > 0:
                # 精化 stage 吃上一级组合出的类别概率（与 mstcn2 的精化输入口径一致）。
                z = self.refine_trunks[index - 1](
                    torch.softmax(compose_log_probability(*pairs[-1]), dim=1)
                )
            pairs.append(head(z))
        return pairs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 推理约定：只取最后一个 stage，转回 [B, T, C]（内容为对数概率）。
        actionness, class_logits = self._forward_stages(x)[-1]
        return compose_log_probability(actionness, class_logits).transpose(1, 2)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """每个 stage 的加权 CE（作用于组合对数概率）+ λ·T-MSE（+ 可选 actionness BCE）。"""

        total = x.new_zeros(())
        for actionness, class_logits in self._forward_stages(x):
            log_p = compose_log_probability(actionness, class_logits)
            # log_p 已归一化，criterion 内部的 log_softmax 是恒等映射，故口径与其它模型一致。
            total = total + criterion(
                log_p.transpose(1, 2).reshape(-1, self.num_classes), y.reshape(-1)
            )
            smooth = torch.clamp(
                (log_p[:, :, 1:] - log_p[:, :, :-1].detach()) ** 2, min=0.0, max=self.tmse_clip ** 2
            ).mean()
            total = total + self.tmse_weight * smooth
            if self.actionness_aux_weight > 0.0:
                target = (y != 0).float().unsqueeze(1)  # [B,1,T]，idle=0、任意动作=1
                aux = F.binary_cross_entropy_with_logits(actionness, target)
                total = total + self.actionness_aux_weight * aux
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
