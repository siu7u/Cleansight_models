"""GRU 时序模型（纯 nn.Module，无监督/喂入语义）。

模型只提供网络结构：输入 ``[B, T, F]``、输出逐帧 logits ``[B, T, C]``。监督口径（末帧
vs 逐帧）与推理方式（滑窗 vs 全序列）由流水线决定，不写在模型里。

单向 GRU 每帧输出只依赖当前帧与历史帧，因此**因果**，可用于滑窗流式推理，也可用于全
序列离线推理。规模（hidden/num_layers）由模型配置表达。

形态 B（``image_dim`` > 0）：输入尾部 ``image_dim`` 维是冻结 backbone 预计算的逐帧图像
embedding，先经线性投影头（``image_proj_dim`` 维）再与 bbox 块拼接送入 GRU——投影头是
形态 B 唯一新增的可训练参数（backbone 零训练，见 docs/features/IMAGE_FEATURE_TRAINING.md §4.1）。
"""

from __future__ import annotations

import torch
import numpy as np
import torch.nn as nn


class GRUClassifier(nn.Module):
    """因果 GRU 时序分类器，输出逐帧动作 logits。

    ``dropout`` 仅作用于 num_layers > 1 时的层间（PyTorch 约定），单层时静默置零；
    默认 0 保持与历史 checkpoint 行为一致，配置 ``model.dropout`` 可开启（配方修复
    推荐 0.2~0.3，缓解小数据过拟合坍缩）。

    ``normalization``（默认 ``none`` / 可选 ``zscore``）为**输入归一化口径**：GRU 历史上直接
    吃原始量纲，而 ROI 契约内部跨两个数量级（presence/count ~0.1~0.7、max_area ~1e-4~1e-2），
    提案 P2a 要求对照 z-score。开启后由 ``fit_normalization(features)`` 按**训练集**逐维统计
    写入 buffer，随 checkpoint 持久化，评估/推理自动复用同一变换；``norm_clip`` 可把标准化
    结果截断到 ±norm_clip（防止近零方差维被放大，见 INPUT_DESIGN_PROPOSAL §1.3）。
    默认 ``none`` 时与历史行为完全一致（不注册 buffer、不声明 provenance）。

    ``image_dim`` 为输入尾部图像 embedding 块宽度（0 = 纯 bbox 契约，不建投影头）；
    ``image_proj_dim`` 为投影后宽度，GRU 实际输入维度 = ``input_dim - image_dim +
    image_proj_dim``。投影逐帧进行，不引入跨帧依赖，因果性不变。
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden=128,
        num_layers=3,
        dropout=0.0,
        image_dim=0,
        image_proj_dim=64,
        normalization="none",
        norm_clip=None,
    ):
        super().__init__()
        self.normalization = str(normalization or "none").lower()
        if self.normalization not in {"none", "zscore"}:
            raise ValueError(
                f"model.normalization 只支持 'none' / 'zscore'，实际 {normalization!r}"
            )
        self.normalization_enabled = self.normalization == "zscore"
        self.norm_clip = float(norm_clip) if norm_clip else 0.0
        if self.normalization_enabled:
            # 逐维统计 [F]，初值直通；随 state_dict 存取，评估时自动应用同一变换。
            self.register_buffer("norm_mean", torch.zeros(int(input_dim)))
            self.register_buffer("norm_std", torch.ones(int(input_dim)))
        self.image_dim = int(image_dim)
        rnn_input_dim = input_dim
        if self.image_dim:
            if not 0 < self.image_dim < input_dim:
                raise ValueError(
                    f"model.image_dim 必须在 (0, input_dim={input_dim}) 内，实际 {image_dim}"
                )
            if int(image_proj_dim) <= 0:
                raise ValueError(f"model.image_proj_dim 必须是正整数，实际 {image_proj_dim}")
            self.image_proj: nn.Module | None = nn.Linear(self.image_dim, int(image_proj_dim))
            rnn_input_dim = input_dim - self.image_dim + int(image_proj_dim)
        else:
            self.image_proj = None
        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, num_classes)

    def normalizer_spec(self) -> str | None:
        """归一化口径的溯源声明；``None`` 表示直通（未归一化），供 checkpoint meta 记录。"""

        if not self.normalization_enabled:
            return None
        clip = f"/clip={self.norm_clip:g}" if self.norm_clip else ""
        return f"zscore/train-set/buffers/v1{clip}"

    def fit_normalization(self, features: list) -> None:
        """训练前钩子：按训练集逐维 z-score 统计写入 buffer（未启用时为空操作）。

        ``features`` 为逐视频 ``[T, F]`` 特征列表；拼接后算逐维 mean/std，std 小于 1e-3 的
        维置 1.0 避免除零（守卫比 MS-TCN 的 1e-4 更严：ROI 契约下曾出现 |z|≈95.6，
        见 ``docs/features/INPUT_DESIGN_PROPOSAL.md`` §1.3）。统计写入 buffer 后随
        checkpoint 持久化，评估与在线推理复用同一变换。
        """

        if not self.normalization_enabled:
            return
        x = np.concatenate(features, axis=0).astype(np.float64)  # [ΣT, F]
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std < 1e-3] = 1.0
        device = self.norm_mean.device
        self.norm_mean.copy_(torch.tensor(mean, dtype=torch.float32, device=device))
        self.norm_std.copy_(torch.tensor(std, dtype=torch.float32, device=device))

    def forward(self, x):
        if self.normalization_enabled:
            x = (x - self.norm_mean) / self.norm_std
            if self.norm_clip:
                x = torch.clamp(x, -self.norm_clip, self.norm_clip)
        if self.image_proj is not None:
            bbox, image = x[..., : -self.image_dim], x[..., -self.image_dim :]
            x = torch.cat([bbox, self.image_proj(image)], dim=-1)  # (B, T, F - image_dim + proj)
        out, _ = self.rnn(x)  # (B, T, H)
        return self.head(out)  # (B, T, num_classes)
