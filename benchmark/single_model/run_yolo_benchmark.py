#!/usr/bin/env python3
"""运行并汇总 CleanSight 分组 YOLO 模型的单模型 benchmark。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "yolo-detection" / "pipeline"
OUT_DIR = ROOT / "benchmark" / "single_model"
LATEST_DIR = OUT_DIR / "latest"
MODEL_CATALOG = ROOT / "model_manager" / "models.yaml"


def build_run_id(version: str | None) -> str:
    """生成用于归档 benchmark summary 的版本化运行编号。"""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not version:
        return timestamp
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", version.strip()).strip("-")
    return f"{slug or 'version'}-{timestamp}"


def load_yolo_model_specs() -> dict[str, dict]:
    """从 `model_manager/models.yaml` 读取 YOLO 模型登记信息。

    benchmark 只使用清单里的稳定模型 id 和分组目标；实际验证仍交给
    `04_validate.py` 执行。
    """

    if not MODEL_CATALOG.exists():
        return {}
    catalog = yaml.safe_load(MODEL_CATALOG.read_text(encoding="utf-8")) or {}
    specs = {}
    for item in catalog.get("models", []):
        if item.get("family") != "yolo":
            continue
        specs[item["id"]] = item
    return specs


def model_id_for_group(specs: dict[str, dict], group: str) -> str | None:
    """返回某个 YOLO 分组在模型清单中的登记 id。"""

    for model_id, item in specs.items():
        if item.get("target") == group:
            return model_id
    return None


def group_for_model_id(specs: dict[str, dict], model_id: str) -> str:
    """把 `yolo.group1_large` 这类模型 id 解析为 YOLO 分组名。"""

    item = specs.get(model_id)
    if not item:
        raise SystemExit(f"未知 YOLO model id: {model_id}")
    return str(item["target"])


def report_path_for(group: str, split: str) -> Path:
    """返回某个分组和 split 对应的当前验证报告路径。"""

    name = "acceptance_report.md" if split == "val" else f"acceptance_report_{split}.md"
    return PIPELINE / "runs" / group / name


def default_checkpoint_for(group: str) -> Path:
    """返回某个 YOLO 分组默认使用的 checkpoint 路径。"""

    return PIPELINE / "runs" / group / "weights" / "best.pt"


def parse_report(path: Path, model_id: str | None, split: str, checkpoint: Path | None) -> dict:
    """解析一份 YOLO 验收 Markdown 报告，提取 benchmark 字段。"""

    text = path.read_text(encoding="utf-8")
    parsed_split = split
    split_match = re.search(r"数据集 split:\s*`([^`]+)`", text)
    if split_match:
        parsed_split = split_match.group(1)

    parsed_checkpoint = str(checkpoint) if checkpoint else None
    ckpt_match = re.search(r"权重:\s*`([^`]+)`", text)
    if ckpt_match:
        parsed_checkpoint = ckpt_match.group(1)

    result = {
        "group": path.parent.name,
        "model_id": model_id,
        "family": "yolo",
        "checkpoint": parsed_checkpoint,
        "dataset": {
            "name": f"datasets/{path.parent.name}",
            "split": parsed_split,
            "format": "ultralytics-yolo",
        },
        "report": str(path.relative_to(ROOT)),
        "status": "PASS" if "PASS" in text else "FAIL" if "FAIL" in text else "UNKNOWN",
        "overall": {},
        "per_class": [],
        "artifacts": {},
        "reasons": [],
    }
    artifact_match = re.search(r"逐图预测:\s*`([^`]+)`", text)
    if artifact_match:
        result["artifacts"]["predictions"] = artifact_match.group(1)

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

    result["metrics"] = {
        "overall": result["overall"],
        "per_class": result["per_class"],
    }
    result["gates"] = {
        "status": result["status"],
        "reasons": result["reasons"],
    }
    return result


def run_validate(groups: list[str], split: str, weights: str | None) -> int:
    """对指定 YOLO 分组调用 `04_validate.py`，并返回退出码。"""

    cmd = [sys.executable, "04_validate.py", *groups, "--split", split]
    if weights:
        if len(groups) != 1:
            raise SystemExit("--weights 只能和单个 YOLO 分组或 --model 一起使用")
        cmd.extend(["--weights", weights])
    proc = subprocess.run(cmd, cwd=PIPELINE)
    return proc.returncode


def collect_groups(requested: list[str]) -> list[str]:
    """解析命令行指定的 YOLO 分组，或从 pipeline 配置读取全部分组。"""

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


def write_summary(
    items: list[dict],
    validate_code: int,
    version: str | None,
    split: str,
    weights: str | None,
) -> tuple[Path, Path]:
    """将 YOLO 单模型 benchmark 汇总写成 JSON 和 Markdown。"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    archive_dir = OUT_DIR / "reports"
    archive_dir.mkdir(parents=True, exist_ok=True)
    run_id = build_run_id(version)
    payload = {
        "schema_version": 1,
        "benchmark": "single_model_yolo",
        "version": version,
        "run_id": run_id,
        "family": "yolo",
        "pipeline": str(PIPELINE.relative_to(ROOT)),
        "dataset": {
            "split": split,
            "format": "ultralytics-yolo",
            "source": "yolo-detection/pipeline/datasets",
        },
        "checkpoint": weights,
        "validate_exit_code": validate_code,
        "groups": items,
        "status": "PASS" if items and all(item["status"] == "PASS" for item in items) else "FAIL",
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    latest_json = LATEST_DIR / "yolo_summary.json"
    archive_json = archive_dir / f"yolo_summary_{run_id}.json"
    latest_json.write_text(json_text, encoding="utf-8")
    archive_json.write_text(json_text, encoding="utf-8")

    lines = [
        "# YOLO 单模型 Benchmark 汇总",
        "",
        f"- 版本：`{version or run_id}`",
        f"- 归档编号：`{run_id}`",
        f"- 流水线：`{PIPELINE.relative_to(ROOT)}`",
        f"- 数据集 split：`{split}`",
        f"- 指定权重：`{weights or '默认 runs/<组>/weights/best.pt'}`",
        f"- 验证退出码：`{validate_code}`",
        "",
        "| 模型 | 组 | 结论 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 报告 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in items:
        overall = item.get("overall", {})
        lines.append(
            "| {model_id} | {group} | {status} | {map50} | {map50_95} | {precision} | {recall} | `{report}` |".format(
                model_id=item.get("model_id") or "未登记",
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

    md_text = "\n".join(lines)
    latest_md = LATEST_DIR / "yolo_summary.md"
    archive_md = archive_dir / f"yolo_summary_{run_id}.md"
    latest_md.write_text(md_text, encoding="utf-8")
    archive_md.write_text(md_text, encoding="utf-8")
    return latest_md, archive_md


def main() -> int:
    """运行或汇总分组检测 checkpoint 的 YOLO 验证报告。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="*", help="只验证指定 YOLO 分组，例如 group1_large")
    parser.add_argument("--model", help="指定模型清单 id,例如 yolo.group1_large")
    parser.add_argument("--skip-run", action="store_true", help="只汇总已有报告，不调用 04_validate.py")
    parser.add_argument("--version", help="为本次 benchmark summary 指定版本名，例如 yolo-large-v2")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="评估数据 split。val 用于训练后验收,test 用于最终 holdout。",
    )
    parser.add_argument("--weights", help="指定单个 YOLO 分组要评估的权重路径")
    args = parser.parse_args()

    if not PIPELINE.exists():
        raise SystemExit(f"缺少 YOLO pipeline: {PIPELINE}")

    specs = load_yolo_model_specs()
    requested_groups = list(args.groups)
    if args.model:
        requested_groups.append(group_for_model_id(specs, args.model))
    groups = collect_groups(requested_groups)
    groups = list(dict.fromkeys(groups))
    if args.weights and len(groups) != 1:
        raise SystemExit("--weights 只能和单个 YOLO 分组或 --model 一起使用")

    validate_code = 0
    if not args.skip_run:
        validate_code = run_validate(groups, args.split, args.weights)

    items = []
    missing = []
    for group in groups:
        model_id = model_id_for_group(specs, group)
        report = report_path_for(group, args.split)
        checkpoint = Path(args.weights) if args.weights else default_checkpoint_for(group)
        if report.exists():
            items.append(parse_report(report, model_id, args.split, checkpoint))
        else:
            missing.append(group)
            items.append(
                {
                    "group": group,
                    "model_id": model_id,
                    "family": "yolo",
                    "checkpoint": str(checkpoint),
                    "dataset": {
                        "name": f"datasets/{group}",
                        "split": args.split,
                        "format": "ultralytics-yolo",
                    },
                    "report": str(report.relative_to(ROOT)),
                    "status": "MISSING",
                    "overall": {},
                    "per_class": [],
                    "artifacts": {},
                    "metrics": {"overall": {}, "per_class": []},
                    "gates": {
                        "status": "MISSING",
                        "reasons": ["缺少验收报告；需要先运行 benchmark 或 04_validate.py。"],
                    },
                    "reasons": ["缺少 acceptance_report.md；需要先完成数据集、权重和验证集。"],
                }
            )

    latest_md, archive_md = write_summary(items, validate_code, args.version, args.split, args.weights)
    print(f"已写入 {latest_md}")
    print(f"已归档 {archive_md}")
    if missing:
        print("缺少报告的组: " + ", ".join(missing))
    return validate_code if validate_code else (2 if missing else 0)


if __name__ == "__main__":
    raise SystemExit(main())
