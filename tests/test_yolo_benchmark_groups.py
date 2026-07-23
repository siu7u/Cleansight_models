"""旧 YOLO benchmark 从 pipeline 真源解析模型分组的快速测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmark.single_model.run_yolo_benchmark import (
    collect_groups,
    group_for_model_id,
    load_pipeline_groups,
    model_id_for_group,
)


class YoloBenchmarkGroupsTest(unittest.TestCase):
    """验证 model-id 约定不再依赖第二份 model catalog。"""

    def test_loads_group_and_class_order_from_pipeline_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text(
                "groups:\n  group1_large: [hand, scope_control_body]\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_pipeline_groups(path),
                {"group1_large": ["hand", "scope_control_body"]},
            )

    def test_model_id_round_trip(self) -> None:
        groups = {"group1_large": ["hand"], "group2_small": ["syringe"]}
        self.assertEqual(model_id_for_group("group1_large"), "yolo.group1_large")
        self.assertEqual(
            group_for_model_id("yolo.group1_large", groups), "group1_large"
        )

    def test_unknown_model_and_group_fail_fast(self) -> None:
        groups = {"group1_large": ["hand"]}
        with self.assertRaisesRegex(SystemExit, "未知 YOLO model id"):
            group_for_model_id("yolo.unknown", groups)
        with self.assertRaisesRegex(SystemExit, "未知 YOLO group"):
            collect_groups(["unknown"], groups)


if __name__ == "__main__":
    unittest.main()
