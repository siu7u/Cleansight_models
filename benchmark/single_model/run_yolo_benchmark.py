#!/usr/bin/env python3
"""Run and summarize YOLO single-model benchmark for grouped CleanSight models."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "yolo-detection" / "pipeline"
OUT_DIR = ROOT / "benchmark" / "single_model"


def parse_report(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    result = {
        "group": path.parent.name,
        "report": str(path.relative_to(ROOT)),
        "status": "PASS" if "PASS" in text else "FAIL" if "FAIL" in text else "UNKNOWN",
        "overall": {},
        "per_class": [],
        "reasons": [],
    }

    for name, key in [
        ("mAP@0.5", "map50"),
        ("mAP@0.5:0.95", "map50_95"),
        ("平均 precision", "precision"),
        ("平均 recall", "recall"),
    ]:
        match = re.search(rf"\|\s*{re.escape(name)}\s*\|\s*([0-9.]+)", text)
        if match:
            result["overall"][key] = float(match.group(1))

    class_row = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|$")
    for line in text.splitlines():
        match = class_row.match(line.strip())
        if not match:
            continue
        label = match.group(1).strip()
        if label in {"类别", "------"}:
            continue
        result["per_class"].append(
            {
                "class": label,
                "precision": float(match.group(2)),
                "recall": float(match.group(3)),
                "map50": float(match.group(4)),
            }
        )

    if "## 未达标项" in text:
        tail = text.split("## 未达标项", 1)[1]
        result["reasons"] = [line[2:].strip() for line in tail.splitlines() if line.startswith("- ")]

    return result


def run_validate(groups: list[str]) -> int:
    cmd = [sys.executable, "04_validate.py", *groups]
    proc = subprocess.run(cmd, cwd=PIPELINE)
    return proc.returncode


def collect_groups(requested: list[str]) -> list[str]:
    if requested:
        return requested
    cfg = PIPELINE / "config.yaml"
    groups = []
    in_groups = False
    for raw in cfg.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith("groups:"):
            in_groups = True
            continue
        if in_groups and line and not line.startswith(" "):
            break
        if in_groups:
            match = re.match(r"\s+([A-Za-z0-9_-]+):", line)
            if match:
                groups.append(match.group(1))
    return groups


def write_summary(items: list[dict], validate_code: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "single_model_yolo",
        "pipeline": str(PIPELINE.relative_to(ROOT)),
        "validate_exit_code": validate_code,
        "groups": items,
    }
    (OUT_DIR / "yolo_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# YOLO 单模型 Benchmark 汇总",
        "",
        f"- 流水线：`{PIPELINE.relative_to(ROOT)}`",
        f"- 验证退出码：`{validate_code}`",
        "",
        "| 组 | 结论 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 报告 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in items:
        overall = item.get("overall", {})
        lines.append(
            "| {group} | {status} | {map50} | {map50_95} | {precision} | {recall} | `{report}` |".format(
                group=item["group"],
                status=item["status"],
                map50=f"{overall.get('map50', 0):.3f}" if "map50" in overall else "待测",
                map50_95=f"{overall.get('map50_95', 0):.3f}" if "map50_95" in overall else "待测",
                precision=f"{overall.get('precision', 0):.3f}" if "precision" in overall else "待测",
                recall=f"{overall.get('recall', 0):.3f}" if "recall" in overall else "待测",
                report=item["report"],
            )
        )

    lines += ["", "## 逐类召回", ""]
    for item in items:
        lines += [f"### {item['group']}", "", "| 类别 | Precision | Recall | mAP@0.5 |", "| --- | ---: | ---: | ---: |"]
        if item["per_class"]:
            for pc in item["per_class"]:
                lines.append(f"| {pc['class']} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['map50']:.3f} |")
        else:
            lines.append("| 待测 | 待测 | 待测 | 待测 |")
        if item["reasons"]:
            lines += ["", "未达标项："]
            lines += [f"- {reason}" for reason in item["reasons"]]
        lines.append("")

    (OUT_DIR / "yolo_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="*", help="只验证指定 YOLO 分组，例如 group1_large")
    parser.add_argument("--skip-run", action="store_true", help="只汇总已有报告，不调用 04_validate.py")
    args = parser.parse_args()

    if not PIPELINE.exists():
        raise SystemExit(f"缺少 YOLO pipeline: {PIPELINE}")

    groups = collect_groups(args.groups)
    validate_code = 0
    if not args.skip_run:
        validate_code = run_validate(groups)

    items = []
    missing = []
    for group in groups:
        report = PIPELINE / "runs" / group / "acceptance_report.md"
        if report.exists():
            items.append(parse_report(report))
        else:
            missing.append(group)
            items.append(
                {
                    "group": group,
                    "report": str(report.relative_to(ROOT)),
                    "status": "MISSING",
                    "overall": {},
                    "per_class": [],
                    "reasons": ["缺少 acceptance_report.md；需要先完成数据集、权重和验证集。"],
                }
            )

    write_summary(items, validate_code)
    print(f"已写入 {OUT_DIR / 'yolo_summary.md'}")
    if missing:
        print("缺少报告的组: " + ", ".join(missing))
    return validate_code if validate_code else (2 if missing else 0)


if __name__ == "__main__":
    raise SystemExit(main())
