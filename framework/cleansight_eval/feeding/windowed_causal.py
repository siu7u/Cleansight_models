"""有界因果窗喂入模式（原 realtime，需求 §8.3）。

逐 tick 只喂当前窗口 ``[1, window, F]``，取最后一帧 logits，经因果平滑
``causal_decision`` 得到稳定预测。前 ``window-1`` 帧为冷启动、填充 idle；每个
视频开始前 reset。迁移自 ``temporal_main.eval_model`` 的流式循环与 benchmark 的
``predict_streaming``。

**训练与实时评估共用这同一喂入模式**：训练侧 ``build_training_dataset`` 按"窗口+末帧"
打包样本（单一真源），评估侧 ``evaluate`` 逐窗推理。二者是同一喂入规格的两个消费者。
"""

from __future__ import annotations

import numpy as np
import torch

from ..tasks.temporal.metrics import causal_decision
from ..tasks.temporal.types import build_dataset
from .base import FeedingResult

IDLE_ID = 0


class WindowedCausalFeeding:
    name = "windowed_causal"
    requires_performance = True

    def __init__(self, min_duration: int = 25):
        self.min_duration = min_duration

    def build_training_dataset(self, features, truths, idx, window):
        """训练侧：训练数据 = 本喂入模式的样本形态（窗口 + 末帧标签）。

        这使"训练用哪种喂入"成为显式选择，而非隐式硬编码；窗口/末帧规格由本模式拥有。
        """
        return build_dataset(features, truths, idx=idx, window=window)

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
