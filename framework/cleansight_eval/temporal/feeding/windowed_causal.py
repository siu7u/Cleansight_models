"""有界因果窗喂入模式（原 realtime，需求 §8.3）。

逐 tick 只喂当前窗口 ``[1, window, F]``，取最后一帧 logits，经因果平滑
``causal_decision`` 得到稳定预测。前 ``window-1`` 帧为冷启动、填充 idle；每个
视频开始前 reset。迁移自 ``temporal_main.eval_model`` 的流式循环与 benchmark 的
``predict_streaming``。

**训练与实时评估共用这同一喂入模式**：``build_datasets`` 按"窗口+末帧"打包样本（单一
真源，训练/评估共用的核心构造），评估侧 ``evaluate`` 逐窗推理。二者是同一喂入规格的两个消费者。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..metrics import causal_decision
from .result import FeedingResult

IDLE_ID = 0


class EndoDataset(Dataset):
    """因果滑窗数据集：样本为 ``[window, F]``，标签为窗口最后一帧类别。

    窗口/末帧规格由本喂入模式拥有（与 ``FullSequenceFeeding`` 的整段+逐帧相对）。
    """

    def __init__(self, features: np.ndarray, labels, window: int = 64):
        self.x = torch.from_numpy(features).float()
        self.y = torch.tensor(labels, dtype=torch.long)
        self.w = window

    def __len__(self):
        return len(self.x) - self.w + 1

    def __getitem__(self, idx):
        x = self.x[idx : idx + self.w]
        y = self.y[idx + self.w - 1]
        return x, y


class WindowedCausalFeeding:
    name = "windowed_causal"
    requires_performance = True
    train_batch_size = None  # None → 用 cfg 的 train.batch_size（窗口样本可批处理）

    def __init__(self, min_duration: int = 25):
        self.min_duration = min_duration

    def build_datasets(self, features, truths, idx, window):
        """按视频构造样本容器（训练与评估共用的核心构造）：窗口 + 末帧标签。

        窗口/末帧规格由本模式拥有；训练侧再 ConcatDataset 拼接、评估侧逐视频消费。
        """
        return [EndoDataset(features[i], truths[i], window) for i in idx]

    def evaluate(self, family, model, datasets, device) -> FeedingResult:
        model.eval()
        video_preds, video_gts = [], []
        with torch.no_grad():
            for ds in datasets:
                total_frames = ds.x.shape[0]
                window = ds.w
                preds = np.zeros(total_frames, dtype=np.int64)
                preds[: window - 1] = IDLE_ID  # 冷启动

                pending, stable, count = None, IDLE_ID, 0  # 每视频 reset
                for i in range(len(ds)):
                    x, _ = ds[i]
                    x = x.unsqueeze(0).to(device)  # [1, window, F]
                    last = family.predict_frame_logits(model, x)
                    pending, stable, count = causal_decision(last, pending, stable, count)
                    preds[i + window - 1] = stable
                video_preds.append(preds)
                video_gts.append(ds.y.numpy())

        window = datasets[0].w if datasets else None
        semantics = {
            "mode": "windowed_causal",
            "sees": "causal_sliding_window",
            "window": window,
            "advance": 1,
            "cold_start": f"前 {window - 1 if window else '?'} 帧填充 idle",
            "reset": "per_video",
            "smoothing": f"causal_decision(min_duration={self.min_duration})",
        }
        return FeedingResult(video_preds, video_gts, semantics)
