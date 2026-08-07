#!/usr/bin/env python3
"""生成逐类详细准确率报告（每个类的 P / R / mAP50）。

读取 runs/aug_compare_*.json，输出：
  - EXPERIMENTS/per_class_report.md   Markdown 汇总
  - EXPERIMENTS/per_class_report.csv  CSV（便于表格/分享）

用法: python tools/report_per_class.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
EXP = REPO / "EXPERIMENTS"

# 类顺序（按 data.yaml 定义）
CLASS_ORDER = {
    "group1_large": ["hand", "scope_control_body", "scope_mid_section"],
    "group2_small": ["syringe", "air_gun", "scope_distal_end", "short_brush", "brush_tip_out"],
}


def load_results() -> list[dict]:
    out = []
    for jf in sorted(RUNS.glob("aug_compare_*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for r in data:
            if "val" not in r or "error" in r:
                continue
            out.append({
                "group": r.get("group", "?"),
                "model": r.get("model", "?"),
                "preset": r.get("preset", "?"),
                "val": r["val"],
            })
    # 补充结果（如被中断的实验补跑 val 得到的逐类数据）
    for jf in sorted(RUNS.glob("*_perclass.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        for r in data:
            if "val" not in r:
                continue
            out.append({
                "group": r.get("group", "?"),
                "model": r.get("model", "?"),
                "preset": r.get("preset", "?"),
                "val": r["val"],
            })
    return out


def group_of(r: dict) -> str:
    return r["group"]


def main() -> None:
    results = load_results()
    if not results:
        raise SystemExit("没有找到结果 JSON（runs/aug_compare_*.json）")

    EXP.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# 逐类详细准确率报告（Per-Class）",
        "",
        f"生成时间: {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
        f"实验数: {len(results)}",
        "",
        "> 每类指标：**mAP50**（主指标）、P（Precision）、R（Recall）。",
        "> 加粗 = 该类在该组所有实验中的最佳 mAP50。",
        "",
    ]
    csv_rows = [["group", "model", "preset", "class", "precision", "recall", "map50"]]

    for group in sorted({r["group"] for r in results}):
        classes = CLASS_ORDER.get(group, [])
        grp = [r for r in results if r["group"] == group]
        md_lines += [f"## {group}", ""]
        # 逐类最佳 mAP50（跨该组所有实验）
        best_map = {}
        for r in grp:
            for c, pc in r["val"].get("per_class", {}).items():
                best_map[c] = max(best_map.get(c, 0.0), pc.get("map50", 0.0))

        header = "| 模型 | 预设 | " + " | ".join(
            f"{c}<br>mAP50 / P / R" for c in classes) + " |"
        sep = "|---|--|" + "---|" * len(classes)
        md_lines += [header, sep]
        for r in sorted(grp, key=lambda x: (x["model"], x["preset"])):
            per = r["val"].get("per_class", {})
            cells = []
            for c in classes:
                pc = per.get(c)
                if pc is None:
                    cells.append("- / - / -")
                    continue
                m, p, rc = pc["map50"], pc["precision"], pc["recall"]
                mark = "**" if m >= best_map.get(c, 0) - 1e-9 else ""
                cells.append(f"{mark}{m:.3f}{mark} / {p:.3f} / {rc:.3f}")
            md_lines.append(f"| {r['model']} | {r['preset']} | " + " | ".join(cells) + " |")
            for c in classes:
                pc = per.get(c)
                if pc is not None:
                    csv_rows.append([group, r["model"], r["preset"], c,
                                     pc["precision"], pc["recall"], pc["map50"]])
        md_lines += ["", f"最佳 mAP50 基准：{ {c: round(v, 3) for c, v in best_map.items()} }", ""]

    (EXP / "per_class_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    with open(EXP / "per_class_report.csv", "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"已生成: {EXP / 'per_class_report.md'}")
    print(f"已生成: {EXP / 'per_class_report.csv'}")
    print("\n" + "\n".join(md_lines))


if __name__ == "__main__":
    main()
