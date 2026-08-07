#!/usr/bin/env python3
"""逐类最优策略分析：对每个 (组, 类) 推荐最佳 模型×增强预设 组合。

数据源：EXPERIMENTS/per_class_report.csv（由 report_per_class.py 生成）。
推荐逻辑：
  1. 主指标 mAP50 最高者为候选；
  2. 若次优与最优差距 < 0.02，用 P/R 平衡（min(P,R) 更大者）与训练成本（yolo11n 更快）决胜；
  3. mAP50 < 0.05 的类标记"检不出"→ 建议淘汰转 ROI 特征融合。

输出：EXPERIMENTS/class_strategy.md
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "EXPERIMENTS"
CSV_PATH = EXP / "per_class_report.csv"

# yolo11n 训练成本约为 yolo11s 的一半
COST_NOTE = {"yolo11s": "~75min/预设", "yolo11n": "~37min/预设"}


def load() -> dict[tuple[str, str], list[dict]]:
    """{(group, class): [ {model, preset, precision, recall, map50}, ... ]}"""
    rows: dict[tuple[str, str], list[dict]] = {}
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["group"], r["class"]), []).append({
                "model": r["model"], "preset": r["preset"],
                "precision": float(r["precision"]), "recall": float(r["recall"]),
                "map50": float(r["map50"]),
            })
    return rows


def pick_best(items: list[dict]) -> dict:
    """在候选里挑最优：mAP50 主指标，差距 <0.005 时看 min(P,R) 与模型成本。"""
    best = max(items, key=lambda x: x["map50"])
    near = [x for x in items if best["map50"] - x["map50"] < 0.005]
    if len(near) > 1:
        # 用 min(P,R)（更平衡者胜），再平局用 yolo11n（更快）
        best = max(near, key=lambda x: (min(x["precision"], x["recall"]),
                                        x["model"] == "yolo11n"))
    return best


def main() -> None:
    rows = load()
    if not rows:
        raise SystemExit(f"没有数据：{CSV_PATH}（先运行 tools/report_per_class.py）")

    lines = [
        "# 逐类最优策略分析（Per-Class Strategy）",
        "",
        f"生成时间: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "> 推荐逻辑：mAP50 主指标；差距 <0.005 时看 P/R 平衡（min(P,R)）与训练成本（yolo11n 更快）。",
        "> mAP50 < 0.05 的类标记 **检不出**，建议淘汰转 ROI 特征融合。",
        "",
    ]

    groups = sorted({g for g, _ in rows})
    for group in groups:
        classes = sorted({c for g, c in rows if g == group},
                         key=lambda c: -max(x["map50"] for x in rows[(group, c)]))
        lines += [f"## {group}", "",
                  "| 类别 | 最优策略 | mAP50 | P / R | 备选 | 说明 |",
                  "|---|---|---:|---|---|---|"]
        for c in classes:
            items = rows[(group, c)]
            best = pick_best(items)
            others = sorted((x for x in items if x is not best),
                            key=lambda x: -x["map50"])[:2]
            alt = "; ".join(f"{x['model']}-{x['preset']}({x['map50']:.3f})" for x in others) or "-"
            if best["map50"] < 0.05:
                note = ("⚠️ **检不出**：mAP50≈0，建议淘汰该类，"
                        "转 ROI 图像特征融合（roi_classification）")
            else:
                note = COST_NOTE.get(best["model"], "")
            lines.append(
                f"| {c} | **{best['model']}-{best['preset']}** | {best['map50']:.3f} | "
                f"{best['precision']:.3f} / {best['recall']:.3f} | {alt} | {note} |")
        lines.append("")

    out = EXP / "class_strategy.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成: {out}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
