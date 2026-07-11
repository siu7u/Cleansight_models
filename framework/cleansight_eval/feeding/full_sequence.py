"""全序列喂入模式（窗口→∞，原 offline，需求 §8.3）。

一次性看到完整序列 ``[1, T, F]``，逐帧取 argmax。参照 benchmark 的
``predict_full_sequence``。全序列喂入不测量实时延迟（§8.4）。
"""

from __future__ import annotations

import numpy as np
import torch

from .base import FeedingResult


class FullSequenceFeeding:
    name = "full_sequence"
    requires_performance = False

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
