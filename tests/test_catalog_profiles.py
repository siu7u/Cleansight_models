"""模型清单 profile 继承与模板展开测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from model_manager.catalog import load_models


class CatalogProfilesTest(unittest.TestCase):
    """只验证 catalog 解析，不触发任何训练或评估脚本。"""

    def _write_catalog(self, payload: dict) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "models.yaml"
        path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
        return path

    def test_profile_merges_model_overrides_and_expands_templates(self) -> None:
        path = self._write_catalog(
            {
                "version": 1,
                "profiles": {
                    "temporal_base": {
                        "family": "temporal",
                        "adapter": "temporal_main",
                        "input": {
                            "feature_mapping": "legacy-20d-v1",
                            "input_dim": 20,
                            "labels": ["Idle"],
                        },
                        "output": {"report": "REPORT.md"},
                        "commands": {
                            "eval": [
                                "tools/eval_temporal_detailed.py",
                                "--repo",
                                "{workdir}",
                                "--model",
                                "{target}",
                                "--checkpoint",
                                "{checkpoint}",
                            ]
                        },
                    }
                },
                "models": [
                    {
                        "id": "temporal.gru",
                        "profile": "temporal_base",
                        "workdir": "temporal-gru",
                        "target": "gru",
                        "input": {"input_dim": 64},
                        "output": {"checkpoint": "registry/gru-v2/model.pt"},
                    }
                ],
            }
        )

        spec = load_models(path)["temporal.gru"]

        self.assertEqual(spec.family, "temporal")
        self.assertEqual(spec.adapter, "temporal_main")
        self.assertEqual(spec.raw["input"]["feature_mapping"], "legacy-20d-v1")
        self.assertEqual(spec.raw["input"]["input_dim"], 64)
        self.assertEqual(spec.raw["output"]["report"], "REPORT.md")
        self.assertEqual(
            spec.raw["commands"]["eval"],
            [
                "tools/eval_temporal_detailed.py",
                "--repo",
                "temporal-gru",
                "--model",
                "gru",
                "--checkpoint",
                "registry/gru-v2/model.pt",
            ],
        )

    def test_unknown_profile_fails_fast(self) -> None:
        path = self._write_catalog(
            {
                "version": 1,
                "models": [
                    {
                        "id": "temporal.gru",
                        "profile": "missing",
                        "workdir": "temporal-gru",
                        "target": "gru",
                    }
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "未知模型 profile"):
            load_models(path)

    def test_duplicate_model_id_still_fails(self) -> None:
        path = self._write_catalog(
            {
                "version": 1,
                "models": [
                    {
                        "id": "temporal.gru",
                        "family": "temporal",
                        "adapter": "temporal_main",
                        "workdir": "temporal-gru",
                        "target": "gru",
                    },
                    {
                        "id": "temporal.gru",
                        "family": "temporal",
                        "adapter": "temporal_main",
                        "workdir": "temporal-gru",
                        "target": "gru",
                    },
                ],
            }
        )

        with self.assertRaisesRegex(ValueError, "模型 id 重复"):
            load_models(path)


if __name__ == "__main__":
    unittest.main()
