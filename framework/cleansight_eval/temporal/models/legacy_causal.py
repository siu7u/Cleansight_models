"""历史 20 维时序 checkpoint 的精确兼容模型。

这些类只保留旧 checkpoint 的网络结构和参数命名，训练、滑窗推理和 checkpoint
加载仍由 framework Pipeline 统一负责。输入均为 ``[B, T, F]``，输出为
``[B, T, C]``；两种模型都只使用当前帧和历史帧，可安全用于流式窗口。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LegacyCausalTCN(nn.Module):
    """兼容 `causal-tcn-v1` 权重键的因果 TCN 分类器。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (64, 64, 64),
    ):
        super().__init__()
        try:
            from pytorch_tcn import TCN
        except ImportError as exc:  # pragma: no cover - 仅未安装可选兼容依赖时触发
            raise ImportError(
                "legacy_causal_tcn_v1 需要 pytorch-tcn>=1.2；请安装 framework requirements"
            ) from exc
        self.tcn = TCN(
            num_inputs=input_dim,
            num_channels=list(hidden_dims),
            kernel_size=3,
            dropout=0.2,
            causal=True,
            use_norm="weight_norm",
            activation="relu",
        )
        self.head = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 ``[B,T,F]`` 转成 TCN 的 ``[B,F,T]``，返回逐帧 logits。"""

        features = self.tcn(x.transpose(1, 2))
        return self.head(features.transpose(1, 2))


class LegacyCausalTransformer(nn.Module):
    """兼容 `causal-transformer-v1` 权重键的因果 Transformer。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """应用上三角因果 mask，确保第 ``t`` 帧不访问未来帧。"""

        sequence_length = x.shape[1]
        projected = self.proj(x)
        causal_mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=x.device,
            ),
            diagonal=1,
        )
        encoded = self.transformer(
            projected,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.head(encoded)
