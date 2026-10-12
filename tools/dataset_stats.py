"""时序数据集分布统计与 manifest 对齐校验（基线分布表）。

用途：`docs/TEMPORAL_DATASET_TRANSFORMATION_PLAN.md` Phase 0 验收项——改造前后
必须有精确的六类分布表格对照；也是 `docs/DATASET_BUILDING_GUIDE.md` §6.5
"统计过 train/val/test 类别分布"的工具化（此前靠人工口述/手工统计）。

对数据集的 ``labels/<split>/``（每行 ``frame_id action_id``）统计：

- 每 split：视频数、每类帧数、每类占比、**缺类列表**（六类不齐显式标出）
- manifest 对齐校验（给了 --manifest-dir 时）：manifest 列出的序列在 labels/ 中
  是否都有文件、labels/ 中是否有 manifest 之外的游离序列（防漏并/防混入）
- 全局汇总 + 每类总帧数

输出人类可读表格；--json 写结构化报告（供 CI/对照）。纯读数据集，不推理。

用法：
    # 默认 auto 数据集（也可 --dataset 指定其他时序数据集根）
    python tools/dataset_stats.py --dataset datasets/cleansight-ActionMixed-auto \
        --manifest-dir benchmark/manifests/actionmixed-auto \
        --json outputs/quality/actionmixed-auto-distribution.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ACTION_CLASSES = [
    "idle", "air_injection", "flush", "long_brush_insert",
    "long_brush_withdraw", "short_brush_cleaning",
]


def _label_frames(label_file: Path) -> dict[int, int]:
    """读单序列动作标签，返回 ``{动作 id: 帧数}``（跳过非两列/非整数行）。"""

    counts: dict[int, int] = {}
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                action = int(parts[1])
            except ValueError:
                continue
            counts[action] = counts.get(action, 0) + 1
    return counts


def build_distribution(
    dataset_root: Path,
    *,
    split_names: tuple[str, ...] = ("train", "val", "test"),
    action_classes: tuple[str, ...] | None = None,
    manifest_dir: Path | None = None,
) -> dict:
    """数据集根 → 分布报告 dict（每 split 明细 + 汇总 + manifest 对齐校验）。"""

    actions = action_classes or tuple(ACTION_CLASSES)
    action_ids = list(range(len(actions)))
    labels_root = dataset_root / "labels"
    if not labels_root.is_dir():
        raise FileNotFoundError(f"数据集缺少 labels/ 目录: {labels_root}")

    splits: dict[str, dict] = {}
    warnings: list[str] = []
    for split in split_names:
        split_dir = labels_root / split
        if not split_dir.is_dir():
            warnings.append(f"labels/ 缺少 split 目录: {split_dir}（按空统计）")
            splits[split] = {
                "videos": 0,
                "frames": 0,
                "per_class_frames": {a: 0 for a in action_ids},
                "per_class_ratio": {a: 0.0 for a in action_ids},
                "missing_classes": list(actions),
                "sequences": [],
            }
            continue
        label_files = sorted(f for f in split_dir.glob("*.txt") if f.name != "data.yaml")
        per_class: dict[int, int] = {}
        for label_file in label_files:
            for action, count in _label_frames(label_file).items():
                per_class[action] = per_class.get(action, 0) + count
        total = sum(per_class.values())
        missing = [action_ids.index(a) for a in action_ids if a not in per_class]
        splits[split] = {
            "videos": len(label_files),
            "frames": total,
            "per_class_frames": {a: per_class.get(a, 0) for a in action_ids},
            "per_class_ratio": {a: (per_class.get(a, 0) / total) if total else 0.0 for a in action_ids},
            "missing_classes": [actions[a] for a in missing],
            "sequences": sorted(f.name[:-4] for f in label_files),
        }

    total_frames = sum(split["frames"] for split in splits.values())
    summary = {
        "videos": sum(split["videos"] for split in splits.values()),
        "frames": total_frames,
        "per_class_frames": {
            a: sum(split["per_class_frames"][a] for split in splits.values()) for a in action_ids
        },
    }

    alignment: dict = {}
    if manifest_dir is not None:
        manifest_dir = Path(manifest_dir)
        alignment["issues"] = []
        for split in split_names:
            if not (labels_root / split).is_dir():
                continue  # split 不存在（按空统计），跳过 manifest 校验
            manifest = manifest_dir / f"{split}.txt"
            if not manifest.is_file():
                alignment["issues"].append(f"{split}: manifest 不存在 {manifest}")
                continue
            registered = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
            actual = splits[split]["sequences"]
            missing = sorted(set(registered) - set(actual))
            extra = sorted(set(actual) - set(registered))
            if missing:
                alignment["issues"].append(f"{split}: manifest 登记但数据缺失 {len(missing)} 个序列: {missing[:5]}")
            if extra:
                alignment["issues"].append(f"{split}: 数据中存在 manifest 未登记序列 {len(extra)} 个: {extra[:5]}")
            alignment[f"{split}_registered"] = len(registered)
            alignment[f"{split}_present"] = len(actual)
        alignment["ok"] = not alignment["issues"]

    return {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "action_classes": list(actions),
        "splits": splits,
        "summary": summary,
        "manifest_alignment": alignment,
        "warnings": warnings,
    }


def print_distribution(report: dict) -> None:
    """把分布报告打印成人类可读表格。"""

    classes = report["action_classes"]
    print("[dataset-stats] ===== 每 split 六类分布 =====")
    for split, info in report["splits"].items():
        cells = " ".join(
            f"{name}:{info['per_class_frames'][i]}"
            for i, name in enumerate(classes)
        )
        print(
            f"[dataset-stats] {split:<5} {info['videos']:>2} 视频 {info['frames']:>6} 帧 | {cells}"
        )
        if info["missing_classes"]:
            print(f"[dataset-stats]   ⚠ 缺类: {', '.join(info['missing_classes'])}")
    summary = report["summary"]
    print("[dataset-stats] ===== 汇总 =====")
    print(
        f"[dataset-stats] 共 {summary['videos']} 视频 {summary['frames']} 帧 | "
        + " ".join(f"{name}:{summary['per_class_frames'][i]}" for i, name in enumerate(classes))
    )
    alignment = report.get("manifest_alignment")
    if alignment:
        print("[dataset-stats] ===== manifest 对齐 =====")
        print(f"[dataset-stats] 状态: {'OK' if alignment['ok'] else '存在问题'}")
        for issue in alignment["issues"]:
            print(f"[dataset-stats]   ⚠ {issue}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="时序数据集六类分布统计 + manifest 对齐校验（基线分布表）"
    )
    p.add_argument("--dataset", default="datasets/cleansight-ActionMixed-auto", help="数据集根（labels/<split>/ 布局）")
    p.add_argument("--manifest-dir", default=None, help="manifest 目录（benchmark/manifests/<testset>/，给定时做对齐校验）")
    p.add_argument("--json", dest="json_out", default=None, help="报告 JSON 输出路径（缺省只打印）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build_distribution(
        Path(args.dataset),
        manifest_dir=Path(args.manifest_dir) if args.manifest_dir else None,
    )
    print_distribution(report)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[dataset-stats] 报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
