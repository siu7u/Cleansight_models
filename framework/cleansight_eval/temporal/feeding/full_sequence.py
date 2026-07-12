"""全序列喂入模式（窗口→∞，原 offline，需求 §8.3）。

一次性看到完整序列 ``[1, T, F]``，逐帧取 argmax。参照 benchmark 的
``predict_full_sequence``。全序列喂入不测量实时延迟（§8.4）。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .result import FeedingResult


class FullSequenceDataset(Dataset):
    """整段序列样本：一条视频一个样本，``x=[T, F]``、``y=[T]``（逐帧标签）。

    与 ``EndoDataset``（窗口 + 末帧标量）相对——离线全序列喂入需要整段 + 逐帧监督。
    变长 T 由 ``train_batch_size=1`` 保证不触发默认 collate 的批内对齐。
    """

    def __init__(self, features: np.ndarray, labels):
        self.x = torch.from_numpy(np.asarray(features)).float()
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.x, self.y


class FullSequenceFeeding:
    name = "full_sequence"
    requires_performance = False
    train_batch_size = 1  # 变长全序列：逐条喂入，不做批内对齐

    def build_datasets(self, features, truths, idx, window):
        """按视频构造样本容器（训练与评估共用的核心构造）：整段序列 + 逐帧标签。

        ``window`` 参数仅为与其他喂入模式签名对称，全序列不加窗、忽略之。
        """
        return [FullSequenceDataset(features[i], truths[i]) for i in idx]

    def evaluate(self, family, model, datasets, device) -> FeedingResult:
        model.eval()
        video_preds, video_gts = [], []
        with torch.no_grad():
            for ds in datasets:
                x = ds.x.unsqueeze(0).to(device)  # [1, T, F]
                logits = family.forward(model, x)  # [1, T, C]
                preds = torch.argmax(logits[0], dim=-1).cpu().numpy().astype(np.int64)
                video_preds.append(preds)
                video_gts.append(ds.y.numpy())
        semantics = {
            "mode": "full_sequence",
            "sees": "full_sequence",
            "windowing": "none",
            "reset": "per_video",
            "note": "全量一次推理，不代表实时行为",
        }
        return FeedingResult(video_preds, video_gts, semantics)
