"""统一 testset manifest 与数据泄漏验证的临时目录测试。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from benchmark.core.testsets import (
    get_testset,
    load_testsets,
    manifest_fingerprint,
    read_split_items,
    validate_catalog,
    validate_spec,
)


ROOT = Path(__file__).resolve().parents[1]


class TestsetManifestTest(unittest.TestCase):
    """只在临时目录构造数据，覆盖正常清单和跨 split 泄漏。"""

    def _write_catalog(self, root: Path, testsets: dict) -> Path:
        path = root / "testsets.yaml"
        path.write_text(
            yaml.safe_dump(
                {"schema_version": 1, "root": ".", "testsets": testsets},
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return path

    def _write_temporal_data(self, root: Path, train: list[str], test: list[str]) -> dict:
        data = root / "endo"
        (data / "splits").mkdir(parents=True)
        (data / "features").mkdir()
        (data / "groundTruth").mkdir()
        (data / "mapping.txt").write_text(
            "0 Idle\n1 Long_Brushing\n2 Short_Brushing\n",
            encoding="utf-8",
        )
        (data / "splits" / "train.bundle").write_text("\n".join(train) + "\n", encoding="utf-8")
        (data / "splits" / "test.bundle").write_text("\n".join(test) + "\n", encoding="utf-8")
        for name in set(train + test):
            (data / "features" / f"{name}.npy").write_bytes(b"npy-placeholder")
            (data / "groundTruth" / f"{name}.txt").write_text("Idle\n", encoding="utf-8")

        common = {
            "family": "temporal",
            "dataset_version": "endo-project-v1",
            "feature_mapping": "legacy-20d-v1",
            "input_dim": 20,
            "labels": ["Idle", "Long_Brushing", "Short_Brushing"],
            "data_root": "endo",
        }
        return {
            "temporal.train": {
                **common,
                "split": "train",
                "manifest": "endo/splits/train.bundle",
                "purpose": "training_only",
                "expected_items": train,
            },
            "temporal.test": {
                **common,
                "split": "test",
                "manifest": "endo/splits/test.bundle",
                "purpose": "locked_holdout_benchmark",
                "expected_items": test,
            },
        }

    def _write_yolo_data(self, root: Path, *, leak: bool) -> Path:
        dataset = root / "yolo"
        for split in ("train", "val", "test"):
            (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "images" / "train" / "video-train.mp4-000001.jpg").touch()
        (dataset / "images" / "val" / "video-val.mp4-000001.jpg").touch()
        test_video = "video-train" if leak else "video-test"
        (dataset / "images" / "test" / f"{test_video}.mp4-000001.jpg").touch()
        data_yaml = dataset / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": {0: "hand"},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return data_yaml

    @staticmethod
    def _yolo_spec() -> dict:
        return {
            "family": "yolo",
            "dataset_version": "detection-v1",
            "split": "test",
            "manifest": "yolo/data.yaml",
            "feature_mapping": "yolo-bbox-v1",
            "input_dim": 3,
            "labels": ["hand"],
            "purpose": "locked_holdout_benchmark",
        }

    def test_temporal_catalog_passes_and_fingerprint_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entries = self._write_temporal_data(root, ["train-a", "train-b"], ["test-a"])
            catalog = load_testsets(self._write_catalog(root, entries))

            self.assertEqual(validate_catalog(catalog), {"temporal.train": [], "temporal.test": []})
            test_spec = get_testset("temporal.test", catalog)
            self.assertEqual(read_split_items(test_spec), ["test-a"])
            self.assertEqual(manifest_fingerprint(test_spec), manifest_fingerprint(test_spec))

    def test_temporal_train_test_overlap_is_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            entries = self._write_temporal_data(root, ["shared"], ["shared"])
            validation = validate_catalog(load_testsets(self._write_catalog(root, entries)))

            self.assertTrue(any("train/test 泄漏" in error for error in validation["temporal.train"]))
            self.assertTrue(any("shared" in error for error in validation["temporal.test"]))

    def test_yolo_clean_split_passes_and_video_leak_fails(self) -> None:
        with tempfile.TemporaryDirectory() as clean_dir:
            clean_root = Path(clean_dir)
            self._write_yolo_data(clean_root, leak=False)
            clean = load_testsets(self._write_catalog(clean_root, {"yolo.test": self._yolo_spec()}))
            self.assertEqual(validate_spec(clean["yolo.test"]), [])

        with tempfile.TemporaryDirectory() as leak_dir:
            leak_root = Path(leak_dir)
            self._write_yolo_data(leak_root, leak=True)
            leaking = load_testsets(self._write_catalog(leak_root, {"yolo.test": self._yolo_spec()}))
            errors = validate_spec(leaking["yolo.test"])
            self.assertTrue(any("train/test" in error and "video-train" in error for error in errors))

    def test_e2e_example_case_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_dir = root / "cases"
            case_dir.mkdir()
            (case_dir / "example.yaml").write_text(
                yaml.safe_dump(
                    {
                        "case_id": "clean-001",
                        "video": "clean-001.mp4",
                        "duration_sec": 180,
                        "expected": {
                            "result": "pass",
                            "required_actions": ["Long_Brushing"],
                            "phases": [
                                {"name": "Long_Brushing", "start_sec": 10, "end_sec": 30}
                            ],
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            entry = {
                "family": "e2e",
                "dataset_version": "e2e-smoke-v1",
                "split": "smoke",
                "manifest": "cases/example.yaml",
                "feature_mapping": "action-timeline-v1",
                "input_dim": None,
                "labels": ["Long_Brushing"],
                "purpose": "schema_smoke_only",
            }
            spec = load_testsets(self._write_catalog(root, {"e2e.smoke": entry}))["e2e.smoke"]
            self.assertEqual(validate_spec(spec), [])
            self.assertEqual(read_split_items(spec), ["clean-001"])

    def test_cli_json_returns_exit_2_for_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_yolo_data(root, leak=True)
            catalog_path = self._write_catalog(root, {"yolo.test": self._yolo_spec()})
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "validate_testsets.py"),
                    "--catalog",
                    str(catalog_path),
                    "--testset",
                    "yolo.test",
                    "--json",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            payload = json.loads(proc.stdout)
            self.assertEqual(proc.returncode, 2)
            self.assertFalse(payload["ok"])
            self.assertTrue(payload["testsets"]["yolo.test"]["errors"])


if __name__ == "__main__":
    unittest.main()
