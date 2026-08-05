"""CLEAN 三种离线时序网络，与后端已验证 checkpoint 的参数结构严格对齐。

三种模型都接收 ``[B,T,F]``，统一返回框架约定的 ``[B,T,C]``。checkpoint 额外保存的
z-score mean/std 通过 ``set_input_normalization`` 注入非持久 buffer，不改变外部
``state_dict`` 的参数键。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _CheckpointNormalizedModel(nn.Module):
    """为外部 checkpoint 提供输入 z-score；统计本身不参与 strict state_dict 契约。"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.register_buffer("_checkpoint_mean", torch.zeros(1, 1, input_dim), persistent=False)
        self.register_buffer("_checkpoint_std", torch.ones(1, 1, input_dim), persistent=False)

    def set_input_normalization(self, mean, std) -> None:
        """注入 checkpoint 保存的 ``[1,F]`` mean/std，维度不符立即拒绝。"""

        mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=self._checkpoint_mean.device)
        std_tensor = torch.as_tensor(std, dtype=torch.float32, device=self._checkpoint_std.device)
        mean_tensor = mean_tensor.reshape(1, 1, -1)
        std_tensor = std_tensor.reshape(1, 1, -1)
        if mean_tensor.shape != self._checkpoint_mean.shape or std_tensor.shape != self._checkpoint_std.shape:
            raise ValueError(
                "checkpoint normalizer 维度不匹配: "
                f"mean={tuple(mean_tensor.shape)}, std={tuple(std_tensor.shape)}, "
                f"expected={tuple(self._checkpoint_mean.shape)}"
            )
        self._checkpoint_mean.copy_(mean_tensor)
        self._checkpoint_std.copy_(torch.where(std_tensor.abs() < 1e-8, torch.ones_like(std_tensor), std_tensor))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num((x - self._checkpoint_mean) / self._checkpoint_std)


class _DilatedResidualLayer(nn.Module):
    """外部 MS-TCN 的膨胀残差块，保持参数命名和 forward 顺序。"""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_dilated(x)
        out = self.act(self.norm(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return self.act(x + out)


class _SingleStageTCN(nn.Module):
    """外部 MS-TCN 的单 stage，输入输出均为 ``[B,C,T]``。"""

    def __init__(self, in_channels: int, classes: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.layers = nn.ModuleList(
            _DilatedResidualLayer(hidden, dilation=2**index, dropout=dropout)
            for index in range(layers)
        )
        self.classifier = nn.Conv1d(hidden, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(x)
        for layer in self.layers:
            z = layer(z)
        return self.classifier(z)


class CleanMSTCNBiLSTM(_CheckpointNormalizedModel):
    """BiLSTM 编码 + MS-TCN 两级 refine；非因果完整序列模型。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 64,
        lstm_layers: int = 2,
        tcn_layers: int = 6,
        refine_stages: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__(input_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden)
        self.bilstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_projection = nn.Conv1d(hidden * 2, hidden, kernel_size=1)
        self.first_stage = _SingleStageTCN(hidden, num_classes, hidden, tcn_layers, dropout)
        self.refine_stages = nn.ModuleList(
            _SingleStageTCN(num_classes, num_classes, hidden, tcn_layers, dropout)
            for _ in range(refine_stages)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.input_projection(self.input_norm(self._normalize(x))))
        z, _ = self.bilstm(z)
        z = self.lstm_projection(z.transpose(1, 2))
        logits = self.first_stage(z)
        for stage in self.refine_stages:
            logits = stage(torch.softmax(logits, dim=1))
        return logits.transpose(1, 2)


def _sinusoidal_position(length: int, dim: int, device) -> torch.Tensor:
    """生成与后端 checkpoint 实现一致的正弦位置编码 ``[T,H]``。"""

    pos = torch.arange(length, device=device).float().unsqueeze(1)
    index = torch.arange(dim, device=device).float().unsqueeze(0)
    divisor = torch.exp(torch.floor(index / 2) * (-math.log(10000.0) / max(dim, 1)))
    encoded = pos * divisor
    output = torch.zeros(length, dim, device=device)
    output[:, 0::2] = torch.sin(encoded[:, 0::2])
    output[:, 1::2] = torch.cos(encoded[:, 1::2])
    return output


class _ASFormerBlock(nn.Module):
    """局部膨胀卷积、多头自注意力和 FFN 组成的 ASFormer block。"""

    def __init__(self, hidden: int, heads: int, dilation: int, dropout: float):
        super().__init__()
        self.local = nn.Conv1d(hidden, hidden, kernel_size=3, padding=dilation, dilation=dilation)
        self.local_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x.transpose(1, 2)).transpose(1, 2)
        x = self.local_norm(x + self.dropout(torch.relu(local)))
        attention, _ = self.attn(x, x, x, need_weights=False)
        x = self.attn_norm(x + self.dropout(attention))
        return self.ffn_norm(x + self.dropout(self.ffn(x)))


class CleanASFormer(_CheckpointNormalizedModel):
    """ASFormer 风格非因果完整序列模型，输入121维业务先验特征。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__(input_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList(
            _ASFormerBlock(hidden, nhead, dilation=2 ** (index % 4), dropout=dropout)
            for index in range(num_layers)
        )
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        z = self.projection(self.input_norm(self._normalize(x)))
        z = z + _sinusoidal_position(time, z.shape[-1], x.device).unsqueeze(0)
        for block in self.blocks:
            z = block(z)
        return self.classifier(z)


class CleanBiGRU(_CheckpointNormalizedModel):
    """三层双向 GRU + 时序卷积头；非因果完整序列模型。"""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden: int = 64,
        num_layers: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__(input_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(
            hidden,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.temporal_head = nn.Sequential(
            nn.Conv1d(hidden * 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.projection(self.input_norm(self._normalize(x))))
        z, _ = self.gru(z)
        return self.temporal_head(z.transpose(1, 2)).transpose(1, 2)
