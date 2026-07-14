"""CARD 只追加历史记录的快速测试。"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from model_manager.history import (
    append_card_record,
    append_evaluation_record,
    append_training_record,
    file_sha256,
)


class HistoryTest(unittest.TestCase):
    """验证 CARD 前缀保持、marker 去重和中文字段渲染。"""

    def test_file_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "artifact.bin"
            payload = b"cleansight\x00model"
            path.write_bytes(payload)
            self.assertEqual(file_sha256(path), hashlib.sha256(payload).hexdigest())

    def test_append_preserves_prefix_and_deduplicates_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            card = Path(temp_dir) / "CARD.md"
            prefix = "# 模型卡\r\n\r\n原始说明保持不变。\r\n".encode("utf-8")
            card.write_bytes(prefix)

            fields = {
                "model": "gru",
                "dataset_path": "data/Endo_Project",
                "checkpoint": "weights/gru-v2/model.pt",
                "input_dim": 20,
            }
            self.assertTrue(append_training_record(card, "train-001", fields))
            first = card.read_bytes()
            self.assertTrue(first.startswith(prefix))
            text = first.decode("utf-8")
            self.assertIn("## 训练历史", text)
            self.assertIn("- 模型: `gru`", text)
            self.assertIn("- 数据集路径: `data/Endo_Project`", text)
            self.assertIn("- 权重: `weights/gru-v2/model.pt`", text)
            self.assertNotIn(str(Path(temp_dir).resolve()), text)

            self.assertFalse(append_training_record(card, "train-001", fields))
            self.assertEqual(card.read_bytes(), first)
            self.assertEqual(text.count("<!-- cleansight-record:训练历史:train-001 -->"), 1)

    def test_same_run_id_can_record_training_and_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            card = Path(temp_dir) / "CARD.md"
            append_training_record(card, {"run_id": "run-007", "epochs": 3, "device": "cpu"})
            append_evaluation_record(
                card,
                {
                    "run_id": "run-007",
                    "split": "test",
                    "metrics": {"f1@0.5": 75.0},
                    "report": "reports/run-007.json",
                },
            )

            text = card.read_text(encoding="utf-8")
            self.assertIn("<!-- cleansight-record:训练历史:run-007 -->", text)
            self.assertIn("<!-- cleansight-record:评估历史:run-007 -->", text)
            self.assertIn("- 训练轮数: `3`", text)
            self.assertIn("- 数据切分: `test`", text)
            self.assertIn("- 评估报告: `reports/run-007.json`", text)

    def test_record_dict_requires_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            card = Path(temp_dir) / "CARD.md"
            with self.assertRaisesRegex(ValueError, "run_id"):
                append_training_record(card, {"model": "gru"})

    def test_custom_section_and_invalid_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            card = Path(temp_dir) / "CARD.md"
            self.assertTrue(append_card_record(card, "发布历史", "release-v1", {"状态": "候选"}))
            self.assertIn("- 状态: `候选`", card.read_text(encoding="utf-8"))
            with self.assertRaises(ValueError):
                append_card_record(card, "训练历史", "bad\nrun", {})


if __name__ == "__main__":
    unittest.main()
