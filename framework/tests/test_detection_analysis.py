"""检测分析（淘汰决策 + 数据集裁剪）逻辑测试。

覆盖：
  - classify_classes 三分组语义（纯函数）
  - build_trimmed_dataset 的 label 过滤与 class id 重映射（临时目录 + 假数据）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.core.analysis import classify_classes
from framework.cleansight_eval.detection.data_tools import build_trimmed_dataset


def _metrics(p, r, note=None):
    m = {"precision": p, "recall": r, "map50": 0.5}
    if note:
        m["note"] = note
    return m


def test_classify_classes_three_way_split():
    per_class = {
        "hand": _metrics(0.85, 0.9),            # keep
        "scope_mid_section": _metrics(0.72, 0.75),  # keep
        "syringe": _metrics(0.2, 0.25),         # borderline (>=0.15)
        "air_gun": _metrics(0.05, 0.1),         # eliminate
        "brush_tip_out": _metrics(0.0, 0.0, "验证集无检出/无样本"),  # eliminate
    }
    keep, borderline, eliminate = classify_classes(per_class, 0.3)
    assert keep == ["hand", "scope_mid_section"]
    assert borderline == ["syringe"]
    assert eliminate == ["air_gun", "brush_tip_out"]


def test_classify_classes_exact_threshold():
    per_class = {
        "a": _metrics(0.3, 0.3),   # 恰好达标 → keep
        "b": _metrics(0.3, 0.1),   # R 低于 0.15 → eliminate
        "c": _metrics(0.2, 0.2),   # 0.15~0.3 → borderline
    }
    keep, borderline, eliminate = classify_classes(per_class, 0.3)
    assert keep == ["a"]
    assert borderline == ["c"]
    assert eliminate == ["b"]


def _write_fake_group(tmp_path: Path) -> Path:
    group_dir = tmp_path / "group2_small"
    (group_dir / "images/train").mkdir(parents=True)
    (group_dir / "images/val").mkdir(parents=True)
    (group_dir / "images/test").mkdir(parents=True)
    (group_dir / "labels/train").mkdir(parents=True)
    (group_dir / "labels/val").mkdir(parents=True)
    (group_dir / "labels/test").mkdir(parents=True)

    (group_dir / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\nnc: 5\nnames:\n"
        "  0: syringe\n  1: air_gun\n  2: scope_distal_end\n  3: short_brush\n  4: brush_tip_out\n",
        encoding="utf-8",
    )
    for split in ("train", "val", "test"):
        (group_dir / "images" / split / "img1.jpg").write_bytes(b"jpeg")
        # 帧1: 保留类(0 syringe) + 淘汰类(1 air_gun)
        (group_dir / "labels" / split / "img1.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n", encoding="utf-8"
        )
        # 帧2: 只含淘汰类(4 brush_tip_out) → 裁剪后应为空 label（不落盘）
        (group_dir / "images" / split / "img2.jpg").write_bytes(b"jpeg")
        (group_dir / "labels" / split / "img2.txt").write_text(
            "4 0.1 0.1 0.05 0.05\n", encoding="utf-8"
        )
    return group_dir


def test_build_trimmed_dataset_filters_and_remaps(tmp_path):
    group_dir = _write_fake_group(tmp_path)
    out_dir = tmp_path / "group2_small_kept"

    yaml_path = build_trimmed_dataset(group_dir, ["syringe", "scope_distal_end", "short_brush"], out_dir)

    import yaml as _yaml
    cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert cfg["nc"] == 3
    assert cfg["names"] == {0: "syringe", 1: "scope_distal_end", 2: "short_brush"}

    for split in ("train", "val", "test"):
        # 图片软链接
        assert (out_dir / "images" / split / "img1.jpg").is_symlink()
        assert (out_dir / "images" / split / "img2.jpg").is_symlink()
        # img1: 只保留 syringe，且 class id 重映射为 0
        kept = (out_dir / "labels" / split / "img1.txt").read_text(encoding="utf-8").strip()
        assert kept == "0 0.5 0.5 0.2 0.2"
        # img2: 只有淘汰类 → 无 label 文件
        assert not (out_dir / "labels" / split / "img2.txt").exists()


def test_build_trimmed_dataset_rejects_empty_keep(tmp_path):
    group_dir = _write_fake_group(tmp_path)
    with pytest.raises(ValueError, match="无保留类别"):
        build_trimmed_dataset(group_dir, [], tmp_path / "empty_out")
