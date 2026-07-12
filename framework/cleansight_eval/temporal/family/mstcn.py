"""MS-TCN 离线分割模型族（离线双向全序列，需求 §13.1 扩展点）。

离线动作分割：一次看到完整序列、双向（非因果）膨胀卷积、逐帧监督。与 GRU（因果滑窗、
末帧监督）共享 temporal 纵的 loader / 指标 / 训练脚手架，差异只落在本族的 forward/loss/
prepare 与 full_sequence 喂入——用多态表达，不硬拆子纵。

网络 vendor 自 ``changhai-offline/segmenter/ms_tcn.py`` 的探索 baseline：用多层膨胀一维卷积
模拟 MS-TCN 的大时间感受野，**是简化版而非论文完整复现**（单 stage、无 T-MSE 平滑）。
每个架构自足于单文件：网络（``MSTCN``）与族契约（``MstcnFamily``）同处此处。

输入归一化（按训练集 z-score）以注册 buffer 内置于网络：随 state_dict 存取、train/eval
自动一致应用、无需另进 checkpoint JSON meta——符合"checkpoint 自描述"不变量（§8.1）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


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
    0/1（直通），由 ``MstcnFamily.prepare`` 在训练前按训练集统计写入。
    """

    def __init__(self, in_dim: int, classes: int, hidden: int = 32):
        super().__init__()
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
        # Conv1d 需要 [B, channels, T]，先把 feature_dim 转到 channel 维。
        z = self.input_projection(x.transpose(1, 2))
        z = self.blocks(z)
        logits = self.classifier(z)  # [B, C, T]
        return logits.transpose(1, 2)  # [B, T, C]


_ARCHS = {"ms_tcn": MSTCN}


class MstcnFamily:
    family_id = "mstcn"

    def build_network(self, model_cfg: dict) -> nn.Module:
        arch = model_cfg.get("arch", "ms_tcn")
        if arch not in _ARCHS:
            raise KeyError(f"未注册的离线分割架构: {arch}；已注册: {sorted(_ARCHS)}")
        return _ARCHS[arch](
            in_dim=model_cfg["input_dim"],
            classes=model_cfg["num_classes"],
            hidden=model_cfg.get("hidden", 32),
        )

    def prepare(self, model: nn.Module, features: list) -> None:
        """训练前钩子：按训练集 z-score 统计写入网络归一化 buffer。

        与 changhai baseline 一致：拼接全部训练特征算逐维 mean/std，std 过小者置 1.0
        避免除零。写入 buffer 后随 checkpoint 存取，评估时自动应用同一归一化。
        """
        x = np.concatenate(features, axis=0)  # [ΣT, F]
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-4] = 1.0
        dev = model.norm_mean.device
        model.norm_mean.copy_(torch.tensor(mean, dtype=torch.float32, device=dev).reshape(1, 1, -1))
        model.norm_std.copy_(torch.tensor(std, dtype=torch.float32, device=dev).reshape(1, 1, -1))

    def forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return model(x)  # [B, T, C]

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor, criterion) -> torch.Tensor:
        # 逐帧监督：整段序列每帧都算 CE（与因果 GRU 的末帧监督相对）。
        # 注：真·MS-TCN++ 前向为 [S, B, C, T] 多 stage，损失应 = Σ 各 stage CE + λ·T-MSE
        # 平滑损失；将来按 logits.dim() 分支扩展。本简化 baseline 为单 stage。
        num_classes = logits.shape[-1]
        return criterion(logits.reshape(-1, num_classes), y.reshape(-1))

    def predict_frame_logits(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # 仅 latency/windowed 路径会用；离线走 full_sequence 不经此。MS-TCN 非因果，
        # 逐窗末帧语义并不代表其真实使用方式，此处仅为契约完整。
        logits = model(x)  # [1, T, C]
        return logits[0, -1]

    def checkpoint_meta(self, model_cfg: dict, feature_schema: dict, extra: dict) -> dict:
        meta = {
            "family": self.family_id,
            "input_dim": model_cfg["input_dim"],
            "num_classes": model_cfg["num_classes"],
            "model": model_cfg,
            "feature_schema": feature_schema,
            "normalizer": "zscore/train-set/buffers/v1",
        }
        meta.update(extra)
        return meta
