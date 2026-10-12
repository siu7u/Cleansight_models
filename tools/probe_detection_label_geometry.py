"""检测数据集「逐类框几何跨 split 一致性」检查：只读标签，不跑模型。

## 为什么需要它

`benchmark/single_model/run_yolo_benchmark.py` 与 sweep 的早停/选点都依赖 **val** 读数。
但如果某个类在 val 上的**标注口径**与 train 不同（框画得大一圈/小一圈），那么即使模型学得
很好，val 上的 IoU 匹配也会整体垮掉，读数会指向"模型不行"这个错误结论。

本工具把三个 split 的逐类框数、框面积中位与分位数并排打出，并给出 val/test 相对 train 的
**面积中位偏离倍数**。偏离明显（例如 4×）就说明该类的 split 口径不可比，其 val 读数不能
作为模型能力证据，也不能用于选点。

用法（仓库根执行）::

    python tools/probe_detection_label_geometry.py --group group2_small
    python tools/probe_detection_label_geometry.py --group group2_small --json tmp/geom.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")


def class_names(data_yaml: Path) -> dict[int, str]:
    """从 data.yaml 读类别表（只解析 ``names:`` 段，避免依赖 ultralytics）。"""

    names: dict[int, str] = {}
    inside = False
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("names:"):
            inside = True
            continue
        if inside:
            if not line.strip() or not line.startswith((" ", "\t")):
                break
            k, _, v = line.strip().partition(":")
            names[int(k)] = v.strip()
    return names


def collect(labels_dir: Path, n_classes: int) -> dict[int, list[float]]:
    """逐类收集框的归一化面积 ``w*h``。"""

    per: dict[int, list[float]] = {i: [] for i in range(n_classes)}
    for path in labels_dir.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split()
            per[int(fields[0])].append(float(fields[3]) * float(fields[4]))
    return per


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", default="group2_small")
    ap.add_argument("--json", default=None, help="可选：把统计写成 JSON")
    args = ap.parse_args()

    group_dir = ROOT / "datasets" / "cleansight-yolo" / args.group
    names = class_names(group_dir / "data.yaml")
    stats: dict[str, dict[int, list[float]]] = {}
    for split in SPLITS:
        labels_dir = group_dir / "labels" / split
        if not labels_dir.is_dir():
            print(f"[WARN] 缺少 split 目录 {labels_dir}，跳过")
            continue
        stats[split] = collect(labels_dir, len(names))
        print(f"\n===== {args.group} / {split} =====")
        print(f"  {'class':<18}{'boxes':>8}{'med_area':>11}{'p10':>10}{'p90':>10}")
        for cid, name in names.items():
            arr = np.asarray(stats[split][cid])
            if not arr.size:
                print(f"  {name:<18}{0:>8}{'-':>11}{'-':>10}{'-':>10}")
                continue
            print(f"  {name:<18}{arr.size:>8}{np.median(arr):>11.5f}"
                  f"{np.percentile(arr, 10):>10.5f}{np.percentile(arr, 90):>10.5f}")

    if "train" not in stats:
        print("[ERROR] 缺少 train split，无法算偏离倍数", file=sys.stderr)
        return 1

    print("\n=== 相对 train 的面积中位偏离倍数（≈1 才说明 split 口径可比）===")
    print(f"  {'class':<18}{'train':>10}{'val':>18}{'test':>18}")
    payload: dict[str, dict] = {}
    for cid, name in names.items():
        tr = stats["train"][cid]
        tr_med = float(np.median(tr)) if tr else float("nan")
        cells, row = "", {}
        for split in ("val", "test"):
            arr = stats.get(split, {}).get(cid, [])
            if not arr or not tr:
                cells += f"{'n/a':>18}"
                row[split] = None
                continue
            med = float(np.median(arr))
            ratio = med / tr_med
            cells += f"{f'{med:.5f} ({ratio:.2f}x)':>18}"
            row[split] = {"median_area": med, "ratio_vs_train": round(ratio, 3), "boxes": len(arr)}
        row["train"] = {"median_area": tr_med if tr else None, "boxes": len(tr)}
        payload[name] = row
        flag = ""
        for split in ("val", "test"):
            r = payload[name].get(split)
            if r and (r["ratio_vs_train"] >= 1.5 or r["ratio_vs_train"] <= 0.67):
                flag = f"  <== {split} 口径异常"
        print(f"  {name:<18}{tr_med:>10.5f}{cells}{flag}")

    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
