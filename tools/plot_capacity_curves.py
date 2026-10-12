"""容量实验汇总图：参数量 vs 段级指标（读 run 产物，不训练）。

用法：python tools/plot_capacity_curves.py [--out docs/mstcn-capacity/figures/capacity_vs_segmental.png]

数据来源与口径：
- 每条曲线内**配方完全相同**（lr / epochs / patience / best_metric），跨曲线不可混读；
- 折线取各容量点的 **seed 中位数**，误差棒为 **跨 seed 最小–最大**（本仓库噪声地板）；
- `mstcn2` 的 val_loss 与 `mstcn` 不同量纲（含多 stage 深监督 + T-MSE 项），因此这里只画 test 段级指标；
- 水平虚线是逐帧线性探针（LDA）下界——"特征 + 后处理"能做到的水平，模型不高于它就说明特征侧到顶。
  该下界取**离线同口径**（逐帧 argmax、无平滑）；因果口径（md=5 + 窗口冷启动）的探针是另一个数
  （edit 50.63），画在同一张图上会严重误导，见文件内 `PROBE_EDIT` 处的注释。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# 曲线定义：标签 → (glob, 过滤条件, 颜色)
SERIES = (
    ("mstcn · default recipe (lr 2e-3, 30 ep)", "runs/capacity-mstcn/h*/mstcn-*", None, "#d62728", "o"),
    ("mstcn · lr 5e-4, 60 ep", "runs/capacity-lr0005-e60/h*/mstcn-*", None, "#1f77b4", "s"),
    ("mstcn · lr 5e-4, 150 ep", "runs/capacity-lr0005-e150/h*/mstcn-*", None, "#2ca02c", "^"),
    ("mstcn2 · lr 5e-4, 60 ep", "runs/mstcn2-cap-*/h*/mstcn2-*", None, "#9467bd", "D"),
)

# 线性探针下界（roi-grid-144）——必须是**与曲线同口径**的那一档：
# 本图所有曲线都是离线 `full_sequence_temporal`（mstcn/mstcn2），其评测是**逐帧 argmax、零平滑**
# （`smoothing_min_duration` 只实现在 `sliding_window_pipeline`，因果 GRU 专用）。因此下界取
# capacity 摘要里自动算出的 `linear-probe（逐帧 LDA，容量无关下界），md=1` 行：edit 17.86 / F1@0.25 5.31。
# 不要用 `docs/FEATURE_STRATEGY_COMPARE.md` 定版 §B 的 50.63 / 33.58——那是**因果口径**
# （GRU、window 冷启动 + md=5 迟滞平滑），与离线曲线不可比。
PROBE_EDIT = 17.86
PROBE_F1_025 = 5.31
# 全 idle 多数类基线（test 帧准确率）：idle 1458 / 2639
ALL_IDLE_ACC = 1458 / 2639 * 100


def read_run(run: Path) -> dict | None:
    evals = sorted((run / "evals").glob("*.evaluation.json"))
    if not evals:
        return None
    data = json.loads(evals[-1].read_text())
    cfg = json.loads((run / "config.resolved.json").read_text())
    meta_path = run / "checkpoints" / "best.pt.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    temporal = data["metrics"]["details"]["temporal"]
    return {
        "params": meta.get("num_params") or data["model"].get("num_params"),
        "edit": temporal["segment"]["edit"] * 100,
        "f1_025": temporal["segment"]["f1_at_iou"]["0.25"] * 100,
        "acc": temporal["frame"]["accuracy"] * 100,
        "cfg": cfg,
    }


def collect(pattern: str) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for run in sorted(REPO.glob(pattern)):
        record = read_run(run)
        if not record or not record["params"]:
            continue
        buckets.setdefault(str(record["params"]), []).append(record)
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(description="容量实验汇总图")
    parser.add_argument("--out", default="docs/mstcn-capacity/figures/capacity_vs_segmental.png")
    args = parser.parse_args()

    cache_dir = Path(tempfile.gettempdir()) / "cleansight-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), constrained_layout=True)
    panels = (("edit", "Segment edit (higher better)", axes[0], PROBE_EDIT, "probe lower bound (offline)"),
              ("f1_025", "Segmental F1@0.25 (higher better)", axes[1], PROBE_F1_025, "probe lower bound (offline)"),
              ("acc", "Frame accuracy (idle-collapse sensitive)", axes[2], ALL_IDLE_ACC, "all-idle baseline"))

    for key, title, axis, reference, reference_label in panels:
        for label, pattern, _, color, marker in SERIES:
            buckets = collect(pattern)
            points = []
            for params in sorted(buckets, key=int):
                values = [item[key] for item in buckets[params]]
                points.append((int(params), statistics.median(values), min(values), max(values), len(values)))
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            low = [p[1] - p[2] for p in points]
            high = [p[3] - p[1] for p in points]
            axis.errorbar(xs, ys, yerr=[low, high], label=label, color=color, marker=marker,
                          markersize=6, linewidth=1.6, capsize=3, alpha=0.9)
            # 标注 seed 数（点数少时才有意义）
            for x, y, _, _, n in points:
                axis.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(4, -10),
                              fontsize=7, color=color)
        axis.axhline(reference, linestyle="--", linewidth=1.2, color="#555555")
        axis.text(0.99, reference, f" {reference_label} {reference:.1f}", transform=axis.get_yaxis_transform(),
                  ha="right", va="bottom", fontsize=8, color="#555555")
        axis.set_xscale("log")
        axis.set_xlabel("parameters (log scale)")
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.25)

    axes[0].legend(fontsize=8, loc="lower right")
    figure.suptitle("MS-TCN capacity study · test segmental metrics vs parameter count "
                    "(median of seeds, whiskers = min–max)", fontsize=12)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=160)
    print(f"[plot] {out}")

    # 同时落一份机读数据，便于报告引用与复核
    data_out = out.with_suffix(".json")
    payload = {}
    for label, pattern, _, _, _ in SERIES:
        buckets = collect(pattern)
        payload[label] = {
            int(params): {
                "n": len(items),
                "edit": statistics.median([i["edit"] for i in items]),
                "f1_025": statistics.median([i["f1_025"] for i in items]),
                "acc": statistics.median([i["acc"] for i in items]),
                "edit_min": min(i["edit"] for i in items),
                "edit_max": max(i["edit"] for i in items),
            }
            for params, items in sorted(buckets.items(), key=lambda kv: int(kv[0]))
        }
    data_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[plot] {data_out}")


if __name__ == "__main__":
    main()
