"""模型代码与权重目录隔离的快速测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from model_manager.layout import LayoutError, scan_layout, validate_layout


class LayoutTest(unittest.TestCase):
    """仅在临时目录中构造布局，不扫描当前仓库的历史权重。"""

    def _touch(self, root: Path, relative: str) -> Path:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_clean_layout_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._touch(root, "model/network.py")
            self._touch(root, "scripts/train.sh")
            self._touch(root, "tools/README.md")
            self._touch(root, "checkpoints/gru/model.pt")
            self._touch(root, "weights/yolo/best.onnx")
            self._touch(root, "runs/exp-1/metrics.json")
            self._touch(root, "registry/gru-v2/pin.yaml")
            self._touch(root, "registry/gru-v2/checkpoint.sha256")

            self.assertEqual(scan_layout(root), [])
            self.assertIsNone(validate_layout(root))

    def test_weight_in_code_directory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._touch(root, "model/assets/model.pt")
            violations = scan_layout(root)
            self.assertEqual([item.rule for item in violations], ["weight_in_code"])
            self.assertEqual(violations[0].path, Path("model/assets/model.pt"))
            with self.assertRaises(LayoutError):
                validate_layout(root)

    def test_python_in_each_weight_directory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("checkpoints", "weights", "runs", "registry"):
                self._touch(root, f"{name}/helper.py")

            violations = scan_layout(root)
            python_paths = {item.path for item in violations if item.rule == "python_in_weight"}
            self.assertEqual(
                python_paths,
                {
                    Path("checkpoints/helper.py"),
                    Path("weights/helper.py"),
                    Path("runs/helper.py"),
                    Path("registry/helper.py"),
                },
            )

    def test_registry_rejects_direct_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._touch(root, "registry/gru-v1/model.pt")

            violations = scan_layout(root)
            self.assertEqual([item.rule for item in violations], ["weight_in_registry"])
            with self.assertRaisesRegex(LayoutError, "registry.*禁止直接存放权重"):
                validate_layout(root)


if __name__ == "__main__":
    unittest.main()
