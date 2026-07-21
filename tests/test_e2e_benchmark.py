from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmark.e2e_3min.run_e2e_benchmark import score_case, write_report


def _case() -> dict:
    return {
        "case_id": "clean-001",
        "video": "clean-001.mp4",
        "duration_sec": 180,
        "expected": {
            "result": "pass",
            "required_actions": ["Long_Brushing", "Short_Brushing"],
            "phases": [
                {"name": "Long_Brushing", "start_sec": 10, "end_sec": 30},
                {"name": "Short_Brushing", "start_sec": 40, "end_sec": 60},
            ],
            "allowed_time_error_sec": 5,
        },
    }


class E2EBenchmarkMetricTests(unittest.TestCase):
    def test_score_case_reports_one_to_one_timeline_metrics(self) -> None:
        prediction = {
            "result": "pass",
            "actions": [
                {"name": "Long_Brushing", "start_sec": 10, "end_sec": 30},
                {"name": "Short_Brushing", "start_sec": 40, "end_sec": 60},
            ],
        }

        score = score_case(_case(), prediction)
        details = score["timeline_metrics"]["details_at_iou"]["0.50"]

        self.assertEqual(score["status"], "PASS")
        self.assertEqual((details["tp"], details["fp"], details["fn"]), (2, 0, 0))
        self.assertEqual(details["precision"], 1.0)
        self.assertEqual(details["recall"], 1.0)
        self.assertEqual(details["f1"], 1.0)
        self.assertEqual(details["mean_matched_iou"], 1.0)
        self.assertEqual(score["phase_errors"][0]["temporal_iou"], 1.0)

    def test_score_case_counts_false_positive_and_false_negative(self) -> None:
        prediction = {
            "result": "pass",
            "actions": [
                {"name": "Long_Brushing", "start_sec": 10, "end_sec": 30},
                {"name": "Extra_Action", "start_sec": 70, "end_sec": 80},
            ],
        }

        score = score_case(_case(), prediction)
        details = score["timeline_metrics"]["details_at_iou"]["0.50"]

        self.assertEqual(score["status"], "FAIL")
        self.assertEqual((details["tp"], details["fp"], details["fn"]), (1, 1, 1))
        self.assertAlmostEqual(details["precision"], 0.5)
        self.assertAlmostEqual(details["recall"], 0.5)
        self.assertAlmostEqual(details["f1"], 0.5)

    def test_phase_gate_cannot_reuse_one_prediction_for_two_truth_segments(self) -> None:
        case = _case()
        case["expected"]["required_actions"] = ["Long_Brushing"]
        case["expected"]["phases"] = [
            {"name": "Long_Brushing", "start_sec": 10, "end_sec": 20},
            {"name": "Long_Brushing", "start_sec": 12, "end_sec": 22},
        ]
        prediction = {
            "result": "pass",
            "actions": [
                {"name": "Long_Brushing", "start_sec": 11, "end_sec": 21},
            ],
        }

        score = score_case(case, prediction)

        self.assertEqual(score["status"], "FAIL")
        self.assertEqual(sum(bool(item["matched"]) for item in score["phase_errors"]), 1)
        details = score["timeline_metrics"]["details_at_iou"]["0.50"]
        self.assertEqual((details["tp"], details["fp"], details["fn"]), (1, 0, 1))

    def test_write_report_includes_timeline_metric_table(self) -> None:
        prediction = {
            "result": "pass",
            "actions": [
                {"name": "Long_Brushing", "start_sec": 10, "end_sec": 30},
                {"name": "Short_Brushing", "start_sec": 40, "end_sec": 60},
            ],
        }
        score = score_case(_case(), prediction)

        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "report.md"
            write_report(_case(), score, out)
            text = out.read_text(encoding="utf-8")

        self.assertIn("## 时间线 IoU / F1", text)
        self.assertIn("| IoU 阈值 | TP | FP | FN | Precision | Recall | F1 |", text)
        self.assertIn("| 0.50 | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 |", text)


if __name__ == "__main__":
    unittest.main()
