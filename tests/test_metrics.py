"""统一时序/端到端指标的边界回归测试。"""

from __future__ import annotations

import unittest

from framework.cleansight_eval.core.metrics import (
    Interval,
    classification_metrics,
    interval_iou,
    match_intervals,
    segments_from_labels,
    temporal_metrics,
    timeline_metrics,
)


class IntervalMetricTests(unittest.TestCase):
    def test_exact_and_touching_intervals(self) -> None:
        truth = Interval("brush", 0, 10)
        self.assertEqual(interval_iou(truth, Interval("brush", 0, 10)), 1.0)
        self.assertEqual(interval_iou(truth, Interval("brush", 10, 20)), 0.0)

    def test_single_frame_segment_keeps_half_open_end(self) -> None:
        self.assertEqual(segments_from_labels([2]), [Interval(2, 0.0, 1.0)])

    def test_duplicate_prediction_counts_as_false_positive(self) -> None:
        truth = [Interval("brush", 0, 10)]
        predictions = [Interval("brush", 0, 10), Interval("brush", 0, 10)]
        matched = match_intervals(predictions, truth, 0.5)
        self.assertEqual((matched.tp, matched.fp, matched.fn), (1, 1, 0))

    def test_wrong_label_never_matches(self) -> None:
        matched = match_intervals(
            [Interval("short", 0, 10)], [Interval("long", 0, 10)], 0.1
        )
        self.assertEqual((matched.tp, matched.fp, matched.fn), (0, 1, 1))

    def test_global_greedy_matching_does_not_lose_a_valid_pair_by_prediction_order(self) -> None:
        truths = [Interval("brush", 0, 10), Interval("brush", 8, 18)]
        predictions = [Interval("brush", 0, 18), Interval("brush", 0, 7)]

        matched = match_intervals(predictions, truths, 0.5)

        self.assertEqual((matched.tp, matched.fp, matched.fn), (2, 0, 0))
        self.assertEqual(
            {(item.prediction_index, item.truth_index) for item in matched.matches},
            {(0, 1), (1, 0)},
        )

    def test_empty_timeline_is_explicit_perfect_absence(self) -> None:
        matched = match_intervals([], [], 0.5).as_metrics()
        self.assertEqual(matched["f1"], 1.0)
        self.assertIsNone(matched["mean_matched_iou"])


class TemporalMetricTests(unittest.TestCase):
    def test_classification_f1_and_iou_use_zero_to_one_ratios(self) -> None:
        result = classification_metrics([0, 1, 0, 1], [0, 1, 1, 1], [0, 1])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["per_class"]["0"]["iou"], 0.5)
        self.assertTrue(0.0 <= result["macro_f1"] <= 1.0)

    def test_video_boundaries_are_not_merged(self) -> None:
        result = temporal_metrics(
            {"a": [1, 1], "b": [1, 1]},
            {"a": [1, 1], "b": [1, 1]},
            labels=[0, 1],
            thresholds=(0.5,),
        )
        details = result["segment"]["details_at_iou"]["0.50"]
        self.assertEqual(details["tp"], 2)
        self.assertEqual(details["f1"], 1.0)

    def test_all_wrong_segments_count_fp_and_fn(self) -> None:
        result = temporal_metrics(
            {"a": [1, 1, 1]},
            {"a": [2, 2, 2]},
            labels=[0, 1, 2],
            thresholds=(0.5,),
        )
        details = result["segment"]["details_at_iou"]["0.50"]
        self.assertEqual((details["tp"], details["fp"], details["fn"]), (0, 1, 1))
        self.assertEqual(details["precision"], 0.0)
        self.assertEqual(details["recall"], 0.0)
        self.assertEqual(details["f1"], 0.0)

    def test_empty_prediction_and_truth_is_perfect_absence(self) -> None:
        result = temporal_metrics(
            {"a": []},
            {"a": []},
            labels=[0, 1],
            thresholds=(0.5,),
        )
        self.assertEqual(result["frame"]["num_frames"], 0)
        self.assertIsNone(result["frame"]["accuracy"])
        self.assertEqual(result["segment"]["details_at_iou"]["0.50"]["f1"], 1.0)

    def test_missing_item_is_rejected_before_metric_aggregation(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing_pred"):
            temporal_metrics({"a": [1]}, {"a": [1], "b": [1]}, labels=[1])

    def test_warmup_and_invalid_predictions_are_excluded(self) -> None:
        result = temporal_metrics(
            {"a": [-1, -1, 1, 1]},
            {"a": [0, 0, 1, 1]},
            labels=[0, 1],
            start_frame=1,
            thresholds=(0.5,),
        )
        self.assertEqual(result["frame"]["num_frames"], 2)
        self.assertEqual(result["frame"]["accuracy"], 1.0)

    def test_timeline_reuses_one_to_one_segment_contract(self) -> None:
        truth = [{"name": "Long", "start_sec": 0, "end_sec": 10}]
        predictions = [
            {"name": "Long", "start_sec": 0, "end_sec": 10},
            {"name": "Long", "start_sec": 0, "end_sec": 10},
        ]
        result = timeline_metrics(predictions, truth, thresholds=(0.5,))
        details = result["details_at_iou"]["0.50"]
        self.assertEqual((details["tp"], details["fp"], details["fn"]), (1, 1, 0))
        self.assertAlmostEqual(result["f1_at_iou"]["0.50"], 2 / 3)
        self.assertEqual(
            result["metric_spec"]["matching"],
            "label-aware one-to-one global-greedy maximum IoU",
        )
        self.assertEqual(result["metric_spec"]["version"], "interval-matching-v2")


class EditSemanticsTests(unittest.TestCase):
    """固化 `edit` 的语义：它是**段标签序列**的 Levenshtein，对时长完全免疫。

    背景（docs/FEATURE_STRATEGY_COMPARE.md 第二十四轮）：edit 常被误读成"段边界质量"。
    实现里 `edit_score` 只取每段标签、丢弃时长，因此以下两条性质是口径的一部分，
    改动它们等于改口径（必须同时 bump spec 版本）。
    """

    LABELS = ["idle", "flush", "insert"]

    def _edit(self, predicted: list[str], truth: list[str]) -> float:
        result = temporal_metrics({"v": predicted}, {"v": truth}, self.LABELS)
        return result["segment"]["edit"]

    def test_duration_changes_do_not_affect_edit(self) -> None:
        truth = ["idle"] * 20 + ["flush"] * 10 + ["idle"] * 20
        same_order = ["idle"] * 38 + ["flush"] * 1 + ["idle"] * 11  # 顺序对、时长全错、位置也不对
        self.assertAlmostEqual(self._edit(same_order, truth), 1.0)

    def test_extra_segment_is_penalized_even_when_frame_accuracy_is_high(self) -> None:
        truth = ["idle"] * 20 + ["flush"] * 10 + ["idle"] * 20
        extra = ["idle"] * 20 + ["flush"] * 10 + ["idle"] * 10 + ["insert"] + ["idle"] * 9
        result = temporal_metrics({"v": extra}, {"v": truth}, self.LABELS)
        self.assertLess(result["segment"]["edit"], 0.7)          # 序列多一段 → 重罚
        self.assertGreater(result["frame"]["accuracy"], 0.95)    # 但逐帧几乎全对

    def test_missing_segment_is_penalized(self) -> None:
        truth = ["idle"] * 10 + ["flush"] * 10 + ["idle"] * 10
        missing = ["idle"] * 30  # 漏掉 flush 段
        self.assertAlmostEqual(self._edit(missing, truth), 1 / 3)


if __name__ == "__main__":
    unittest.main()
