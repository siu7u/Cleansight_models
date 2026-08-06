#!/usr/bin/env python3
"""汇总 aug_experiments.py 的结果 JSON 并输出对比报告。

用法: python tools/report_aug_results.py [结果json...]
     不带参数时扫描 ~/cleansight-runs/aug_compare_*.json 最新一个。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HOME = Path.home()
RUNS = HOME / "cleansight-runs"


def load_latest() -> Path:
    files = sorted(RUNS.glob("aug_compare_*.json"))
    if not files:
        raise SystemExit(f"没有找到结果文件: {RUNS}/aug_compare_*.json")
    return files[-1]


def report(path: Path) -> None:
    results = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n结果文件: {path}\n")
    print(f"{'预设':<14} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} {'训练耗时':>8}  备注")
    print("-" * 78)
    for r in results:
        if "error" in r:
            print(f"{r.get('preset','?'):<14} ERROR: {r['error'][:60]}")
            continue
        if "val" not in r:
            print(f"{r.get('preset','?'):<14} 未完成")
            continue
        v = r["val"]
        mins = r.get("train_seconds", 0) / 60
        note = f"{r.get('model','?')} imgsz={r.get('imgsz','?')} e={r.get('epochs','?')}"
        print(f"{r['preset']:<14} {v['map50']:>8.4f} {v['map50_95']:>10.4f} "
              f"{v['precision']:>8.4f} {v['recall']:>8.4f} {mins:>7.1f}m  {note}")
        if v.get("per_class"):
            print("  逐类: " + ", ".join(
                f"{c}={pc['map50']:.3f}(P{pc['precision']:.2f}/R{pc['recall']:.2f})"
                for c, pc in v["per_class"].items()))

    valid = [r for r in results if "val" in r and "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["val"]["map50"])
        print(f"\n🏆 最佳 mAP50: {best['preset']} ({best['val']['map50']:.4f})")
        best_p = max(valid, key=lambda r: r["val"]["precision"])
        best_r = max(valid, key=lambda r: r["val"]["recall"])
        print(f"🏆 最佳 Precision: {best_p['preset']} ({best_p['val']['precision']:.4f})")
        print(f"🏆 最佳 Recall: {best_r['preset']} ({best_r['val']['recall']:.4f})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            report(Path(p))
    else:
        report(load_latest())
