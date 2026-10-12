"""框集合编码器 + MS-TCN++ 主干（``model.type = boxset_mstcn2``）。

**动机**：手工 ROI 直方图（`roi-grid-v1..v5`）把每帧压缩成"(检测类, 区域) 的计数与极值"，
**丢掉了框与框之间的关系**。本模型改为直接吃**原始框集合**（`actionmixed-boxset-v1`：
K=8 槽 × [one-hot(8) | cx | cy | w | h]），先用一个**可学习、置换不变**的编码器把每帧的
变长框集合压成一个定长表示，再交给 MS-TCN++ 做时序建模。

**置换不变性怎么保证**：编码器对每个槽**共享同一个 Linear**，再对槽维做**掩码 max/mean 池化**
（掩码由槽内 one-hot 是否全零推出）。因此槽的顺序不影响输出。

**与现有实现的关系**：时序主干**完全复用** `MSTCN2`（多 stage 深监督 + T-MSE + 归一化 buffer），
本类只前置一个编码器并转发 `forward` / `compute_loss`，避免复制一份时序配方。

**归一化**：本模型**不实现** `fit_normalization`（输入的 one-hot 部分不应做 z-score，
坐标缩放交给编码器的 Linear 学），因此流水线不会给它写归一化 buffer。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mstcn2 import MSTCN2


class SlotSetEncoder(nn.Module):
    """逐槽共享 MLP + 掩码 max/mean 池化 → 每帧定长表示（置换不变）。

    输入 ``[B, T, K * slot_dim]``，输出 ``[B, T, out_dim]``。
    槽是否有效由槽内 one-hot 之和是否 > 0 判定（空槽全零）。
    """

    def __init__(self, input_dim: int, num_slots: int, slot_dim: int, out_dim: int):
        super().__init__()
        if input_dim != num_slots * slot_dim:
            raise ValueError(
                f"input_dim({input_dim}) 必须等于 num_slots({num_slots}) × slot_dim({slot_dim})"
            )
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_classes = slot_dim - 4
        self.embed = nn.Sequential(
            nn.Linear(slot_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )
        self.project = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        slots = x.reshape(b, t, self.num_slots, self.slot_dim)
        onehot = slots[..., : self.n_classes]
        valid = onehot.sum(dim=-1) > 0                      # [B, T, K]
        h = self.embed(slots)                               # [B, T, K, out]
        neg_inf = torch.finfo(h.dtype).min
        masked = h.masked_fill(~valid.unsqueeze(-1), neg_inf)
        pooled_max = masked.amax(dim=2)                     # [B, T, out]
        # 整帧无框时 amax 会是 -inf，用 0 兜底避免污染后续层
        empty = ~valid.any(dim=-1, keepdim=True)            # [B, T, 1]
        pooled_max = torch.where(empty, torch.zeros_like(pooled_max), pooled_max)
        weights = valid.unsqueeze(-1).to(h.dtype)
        pooled_mean = (h * weights).sum(dim=2) / weights.sum(dim=2).clamp(min=1.0)
        return self.project(torch.cat([pooled_max, pooled_mean], dim=-1))


class BoxSetMSTCN2(nn.Module):
    """框集合编码器（每帧） + MS-TCN++（跨帧）的串联模型。

    ``forward`` 输入 ``[B, T, K*slot_dim]``、输出 ``[B, T, num_classes]``（仅最后一个 stage）；
    训练损失经 ``compute_loss`` 转发给内部 MS-TCN++（多 stage 深监督 + T-MSE）。
    **双向（非因果）**，只适用于全序列离线流水线。
    """

    def __init__(
        self,
        input_dim: int,
        classes: int,
        hidden: int = 128,
        num_stages: int = 4,
        num_layers: int = 10,
        dropout: float = 0.3,
        tmse_weight: float = 0.15,
        tmse_clip: float = 4.0,
        num_slots: int = 8,
        slot_dim: int = 12,
        slot_embed: int = 64,
        sequence_normalization: str = "none",
    ):
        super().__init__()
        self.num_classes = classes
        self.encoder = SlotSetEncoder(input_dim, num_slots, slot_dim, slot_embed)
        self.core = MSTCN2(
            in_dim=slot_embed,
            classes=classes,
            hidden=hidden,
            num_stages=num_stages,
            num_layers=num_layers,
            dropout=dropout,
            tmse_weight=tmse_weight,
            tmse_clip=tmse_clip,
            sequence_normalization=sequence_normalization,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(self.encoder(x))

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """训练配方转发：先编码，再走 MS-TCN++ 的多 stage 深监督 + T-MSE。"""

        return self.core.compute_loss(self.encoder(x), y, criterion)
