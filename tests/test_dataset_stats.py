"""tools/dataset_stats.py：数据集分布统计与 manifest 对齐校验测试。

验收标准（verification-first）：
- 每 split 六类帧数/占比/缺类标记正确
- manifest 对齐：登记缺失、游离序列两类问题都能检出
- 汇总计数正确；--json 写报告；缺 labels/ 目录报错
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.dataset_stats import build_distribution, main, parse_args  # noqa: E402


def _make_dataset(root: Path, splits: dict, manifest: dict | None = None) -> Path:
    """构造数据集：``splits`` 为 ``{split: {序列: {动作id: 帧数}}}``。"""

    for split, sequences in splits.items():
        for seq, actions in sequences.items():
            label_file = root / "labels" / split / f"{seq}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            frame = 1
            for action, count in actions.items():
                for _ in range(count):
                    lines.append(f"{frame} {action}")
                    frame += 1
            label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if manifest:
        manifest_dir = root / "manifests"
        for split, names in manifest.items():
            path = manifest_dir / f"{split}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return root


class TestDistribution:
    def test_counts_ratios_and_missing(self, tmp_path):
        root = _make_dataset(
            tmp_path,
            {"train": {"a.mp4": {0: 4, 1: 1}}, "val": {"b.mp4": {0: 2}}},
        )
        report = build_distribution(root)
        train = report["splits"]["train"]
        assert train["videos"] == 1 and train["frames"] == 5
        assert train["per_class_frames"] == {0: 4, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0}
        assert train["per_class_ratio"][0] == pytest.approx(0.8)
        assert train["missing_classes"] == [
            "flush", "long_brush_insert", "long_brush_withdraw", "short_brush_cleaning",
        ]
        val = report["splits"]["val"]
        assert val["missing_classes"] == [
            "air_injection", "flush", "long_brush_insert", "long_brush_withdraw",
            "short_brush_cleaning",
        ]
        assert report["summary"]["frames"] == 7

    def test_manifest_alignment_issues(self, tmp_path):
        root = _make_dataset(
            tmp_path,
            {"train": {"a.mp4": {0: 1}, "extra.mp4": {0: 1}}},
            manifest={"train": ["a.mp4", "missing.mp4"]},
        )
        report = build_distribution(root, manifest_dir=root / "manifests")
        alignment = report["manifest_alignment"]
        assert alignment["ok"] is False
        issues = "\n".join(alignment["issues"])
        assert "missing.mp4" in issues and "extra.mp4" in issues

    def test_manifest_alignment_ok(self, tmp_path):
        root = _make_dataset(
            tmp_path,
            {"train": {"a.mp4": {0: 1}}},
            manifest={"train": ["a.mp4"]},
        )
        report = build_distribution(root, manifest_dir=root / "manifests")
        assert report["manifest_alignment"]["ok"] is True

    def test_missing_labels_dir_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="labels/"):
            build_distribution(tmp_path / "nope")


class TestCli:
    def test_parse_args(self):
        args = parse_args(["--dataset", "ds", "--manifest-dir", "m", "--json", "r.json"])
        assert args.dataset == "ds" and args.manifest_dir == "m"

    def test_main_writes_json(self, tmp_path):
        root = _make_dataset(tmp_path, {"train": {"a.mp4": {0: 1}}})
        out = tmp_path / "report.json"
        code = main(["--dataset", str(root), "--json", str(out)])
        assert code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["summary"]["videos"] == 1
