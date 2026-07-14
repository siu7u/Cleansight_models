"""时序评估 artifact 的边界保持测试。"""

from __future__ import annotations

import unittest

try:
    import numpy as np
    import torch
except ModuleNotFoundError:  # pragma: no cover - 轻量 CI 环境可不安装训练依赖
    np = None
    torch = None

if np is not None and torch is not None:
    from benchmark.core.artifacts import temporal_metrics_from_prediction_artifact
    from benchmark.core.temporal_data import TemporalItem
    from tools.eval_temporal_detailed import (
        _predict_item_last_frame,
        build_predictions_artifact,
    )


if torch is not None:
    class ConstantTemporalModel(torch.nn.Module):
        """测试用模型：无论输入窗口内容如何，末帧都预测类别 1。"""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros((x.shape[0], x.shape[1], 2), dtype=torch.float32, device=x.device)
            logits[:, :, 1] = 1.0
            return logits
else:
    class ConstantTemporalModel:  # pragma: no cover - 仅用于缺依赖环境的类名占位
        pass


@unittest.skipIf(np is None or torch is None, "需要 numpy 和 torch")
class TemporalEvalArtifactTest(unittest.TestCase):
    def test_last_frame_prediction_keeps_item_alignment(self) -> None:
        item = TemporalItem(
            name="video-a",
            features=np.zeros((4, 3), dtype=np.float32),
            labels=np.asarray([0, 1, 1, 0], dtype=np.int64),
        )

        predictions, truths = _predict_item_last_frame(
            ConstantTemporalModel(),
            item,
            window=2,
            batch_size=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(predictions, [1, 1, 1])
        self.assertEqual(truths, [1, 1, 0])

    def test_predictions_artifact_preserves_video_boundaries(self) -> None:
        artifact = build_predictions_artifact(
            pred_by_item={"video-a": [1, 1], "video-b": [0]},
            truth_by_item={"video-a": [1, 0], "video-b": [0]},
            index_to_action={0: "Idle", 1: "Long_Brushing"},
            window=64,
            inference_mode="raw_last_frame",
        )

        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(set(artifact["items"]), {"video-a", "video-b"})
        self.assertEqual(artifact["items"]["video-a"]["prediction_start_frame"], 63)
        self.assertEqual(artifact["items"]["video-a"]["predicted_labels"], ["Long_Brushing", "Long_Brushing"])
        self.assertEqual(artifact["items"]["video-b"]["truth_labels"], ["Idle"])

    def test_metrics_can_be_recomputed_from_artifact(self) -> None:
        artifact = build_predictions_artifact(
            pred_by_item={"video-a": [1, 1], "video-b": [1, 1]},
            truth_by_item={"video-a": [1, 1], "video-b": [1, 1]},
            index_to_action={0: "Idle", 1: "Long_Brushing"},
            window=64,
            inference_mode="raw_last_frame",
        )

        metrics = temporal_metrics_from_prediction_artifact(artifact, thresholds=(0.5,))

        self.assertEqual(metrics["segment"]["num_items"], 2)
        self.assertEqual(metrics["segment"]["details_at_iou"]["0.50"]["tp"], 2)


if __name__ == "__main__":
    unittest.main()
