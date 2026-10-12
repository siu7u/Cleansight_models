"""MS-TCN 离线分割模型（纯 nn.Module + 自带输入归一化）。

简化版 MS-TCN：多层膨胀一维卷积模拟大时间感受野（单 stage、无 T-MSE 平滑，非论文完整
复现）。**双向（非因果）**，只适用于全序列离线流水线；滑窗流水线会因 ``causal=False``
拒绝它。网络 vendor 自 ``changhai-offline/segmenter/ms_tcn.py`` 的探索 baseline。

输入归一化（按训练集 z-score）以注册 buffer 内置于网络：随 state_dict 存取、train/eval
自动一致应用、无需另进 checkpoint JSON meta——符合"checkpoint 自描述"不变量。统计由
``fit_normalization`` 在训练前按训练数据写入（流水线以 duck-type 可选钩子调用）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


SEQUENCE_NORMALIZATION_MODES = ("none", "demean")


def _resolve_sequence_normalization(value) -> str:
    """校验 ``sequence_normalization`` 口径：``none``（默认）或 ``demean``（按视频去均值）。

    ``demean`` 在每个序列（一条视频一次前向）内部减去**该序列自身**的逐维均值，用于消掉
    批次级的电平/构图差异（第十三轮探针：它把最差类 flush 的 train→test 可迁移性从 0.281
    修到 0.604）。它需要看到整段序列，因此只适用于全序列离线模型。
    """

    mode = str(value or "none").lower()
    if mode not in SEQUENCE_NORMALIZATION_MODES:
        raise ValueError(
            f"model.sequence_normalization 只支持 {list(SEQUENCE_NORMALIZATION_MODES)}，实际 {value!r}"
        )
    return mode


def _apply_sequence_normalization(x: torch.Tensor, mode: str) -> torch.Tensor:
    """按口径对 ``[B, T, F]`` 输入做序列级变换（``none`` 直通）。"""

    if mode == "demean":
        return x - x.mean(dim=1, keepdim=True)
    return x


class ResidualTemporalBlock(nn.Module):
    """残差膨胀卷积块：在时间轴上扩大上下文。"""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.08):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.net(x))


class MSTCN(nn.Module):
    """简化 MS-TCN：双向膨胀卷积 + 内置输入 z-score 归一化。

    输入 ``[B, T, F]``，输出 ``[B, T, C]``（已转置为框架逐帧约定）。归一化 buffer 初值
    0/1（直通），由 ``fit_normalization`` 在训练前按训练集统计写入。
    """

    def __init__(self, in_dim: int, classes: int, hidden: int = 32,
                 sequence_normalization: str = "none"):
        super().__init__()
        self.sequence_normalization = _resolve_sequence_normalization(sequence_normalization)
        self.input_projection = nn.Conv1d(in_dim, hidden, kernel_size=1)
        self.blocks = nn.Sequential(
            *(ResidualTemporalBlock(hidden, dilation) for dilation in [1, 2, 4, 8, 16, 1, 2, 4])
        )
        self.classifier = nn.Conv1d(hidden, classes, kernel_size=1)
        # 归一化统计随 state_dict 持久化：[1, 1, F]，初值直通。
        self.register_buffer("norm_mean", torch.zeros(1, 1, in_dim))
        self.register_buffer("norm_std", torch.ones(1, 1, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.norm_mean) / self.norm_std
        x = _apply_sequence_normalization(x, self.sequence_normalization)
        # Conv1d 需要 [B, channels, T]，先把 feature_dim 转到 channel 维。
        z = self.input_projection(x.transpose(1, 2))
        z = self.blocks(z)
        logits = self.classifier(z)  # [B, C, T]
        return logits.transpose(1, 2)  # [B, T, C]

    def normalizer_spec(self) -> str:
        """归一化口径的溯源声明（本模型恒按训练集 z-score 归一化）。"""

        return ("zscore/train-set/buffers/v1" if self.sequence_normalization == "none"
                else f"zscore/train-set/buffers/v1+sequence-{self.sequence_normalization}/v1")

    def fit_normalization(self, features: list) -> None:
        """训练前可选钩子：按训练集 z-score 统计写入归一化 buffer。

        与 changhai baseline 一致：拼接全部训练特征算逐维 mean/std，std 过小者置 1.0
        避免除零。写入 buffer 后随 checkpoint 存取，评估时自动应用同一归一化。
        """
        x = np.concatenate(features, axis=0)  # [ΣT, F]
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-4] = 1.0
        dev = self.norm_mean.device
        self.norm_mean.copy_(torch.tensor(mean, dtype=torch.float32, device=dev).reshape(1, 1, -1))
        self.norm_std.copy_(torch.tensor(std, dtype=torch.float32, device=dev).reshape(1, 1, -1))
