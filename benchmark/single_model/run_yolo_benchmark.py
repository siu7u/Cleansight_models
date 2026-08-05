#!/usr/bin/env python3
"""通过统一 framework→benchmark 主链路批量评测分组 YOLO 权重。"""

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
OUT_DIR = ROOT / "benchmark" / "single_model"
LATEST_DIR = OUT_DIR / "latest"
GROUP_CONFIGS = {
    "group1_large": {
        "config": "framework/experiments/yolo-clean-large.yaml",
        "registry": "registry/detection/yolo-group1-large-v1",
    },
    "group2_small": {
        "config": "framework/experiments/yolo-clean-small.yaml",
        "registry": "registry/detection/yolo-group2-small-v1",
    },
}


def build_run_id(version: str | None) -> str:
    """生成稳定归档编号。"""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not version:
        return timestamp
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", version.strip()).strip("-")
    return f"{slug or 'version'}-{timestamp}"


def load_pipeline_groups(path: Path | None = None) -> dict[str, list[str]]:
    """读取分组类别；显式 path 兼容旧 groups YAML，默认读取顶层 registry。"""

    if path is not None:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_groups = payload.get("groups")
        if not isinstance(raw_groups, dict) or not raw_groups:
            raise ValueError(f"{path} 缺少非空 groups mapping")
        return {
            str(group): [str(label) for label in labels]
            for group, labels in raw_groups.items()
        }

    groups: dict[str, list[str]] = {}
    for group, item in GROUP_CONFIGS.items():
        classes_path = ROOT / item["registry"] / "classes.yaml"
        payload = yaml.safe_load(classes_path.read_text(encoding="utf-8")) or {}
        classes = payload.get("classes")
        if not isinstance(classes, dict) or not classes:
            raise ValueError(f"{classes_path} 缺少 classes mapping")
        groups[group] = [str(classes[key]) for key in sorted(classes, key=int)]
    return groups


def model_id_for_group(group: str) -> str:
    """按稳定约定返回 YOLO 模型 ID。"""

    return f"yolo.{group}"


def group_for_model_id(model_id: str, groups: dict[str, list[str]]) -> str:
    """把 ``yolo.<group>`` 解析为已登记分组。"""

    prefix = "yolo."
    group = model_id[len(prefix) :] if model_id.startswith(prefix) else ""
    if group not in groups:
        choices = ", ".join(model_id_for_group(name) for name in sorted(groups))
        raise SystemExit(f"未知 YOLO model id: {model_id}；可用: {choices}")
    return group


def collect_groups(requested: list[str], groups: dict[str, list[str]]) -> list[str]:
    """验证显式分组；未指定时返回全部 registry 分组。"""

    selected = requested or list(groups)
    unknown = [name for name in selected if name not in groups]
    if unknown:
        raise SystemExit(
            f"未知 YOLO group: {', '.join(unknown)}；可用: {', '.join(sorted(groups))}"
        )
    return selected


def _parse_weights(items: list[str]) -> dict[str, Path]:
    """解析可重复的 ``--weights group=/path/best.pt`` 参数。"""

    result: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit("--weights 必须使用 group=/path/to/best.pt")
        group, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"YOLO 权重不存在: {path}")
        result[group] = path
    return result


def _run_group(group: str, checkpoint: Path) -> dict:
    """调用统一 eval CLI 并读取生成的 EvaluationResult。"""

    item = GROUP_CONFIGS[group]
    invocation_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = LATEST_DIR / group / invocation_id
    command = [
        sys.executable,
        "-m",
        "benchmark.cli.eval",
        "--config",
        str(ROOT / item["config"]),
        "--ckpt",
        str(checkpoint),
        "--out-dir",
        str(output_dir),
    ]
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    evaluations = sorted(
        output_dir.glob("*.evaluation.json"),
        key=lambda path: path.stat().st_mtime,
    )
    evaluation_path = evaluations[-1] if evaluations else None
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path is not None
        else None
    )
    return {
        "group": group,
        "model_id": model_id_for_group(group),
        "checkpoint": str(checkpoint),
        "config": item["config"],
        "evaluation": evaluation,
        "evaluation_path": (
            str(evaluation_path.relative_to(ROOT)) if evaluation_path is not None else None
        ),
        "exit_code": proc.returncode,
        "log_tail": (proc.stdout + proc.stderr)[-3000:],
    }


def _metric(item: dict, name: str) -> str:
    metric = (
        ((item.get("evaluation") or {}).get("metrics") or {}).get("summary") or {}
    ).get(name, {})
    if metric.get("state") != "computed":
        return metric.get("state", "MISSING")
    value = metric.get("value")
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else str(value)


def write_summary(results: list[dict], version: str | None) -> tuple[Path, Path]:
    """汇总 EvaluationResult；不解析历史 Markdown 验收报告。"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    reports = OUT_DIR / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    run_id = build_run_id(version)
    payload = {
        "benchmark": "single_model_yolo",
        "version": version,
        "run_id": run_id,
        "groups": results,
    }
    latest_json = LATEST_DIR / "yolo_summary.json"
    archive_json = reports / f"yolo_summary_{run_id}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    latest_json.write_text(text, encoding="utf-8")
    archive_json.write_text(text, encoding="utf-8")

    lines = [
        "# YOLO 单模型 Benchmark 汇总",
        "",
        f"- 版本：`{version or run_id}`",
        "- 训练与推理：`framework`",
        "- 指标与产物：`benchmark`",
        "",
        "| 模型 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 结果 | 状态 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['model_id']} | {_metric(item, 'map50')} | {_metric(item, 'map50_95')} | "
            f"{_metric(item, 'precision')} | {_metric(item, 'recall')} | "
            f"`{item.get('evaluation_path') or '未生成'}` | "
            f"{'OK' if item['exit_code'] == 0 else 'CHECK'} |"
        )
    latest_md = LATEST_DIR / "yolo_summary.md"
    archive_md = reports / f"yolo_summary_{run_id}.md"
    markdown = "\n".join(lines) + "\n"
    latest_md.write_text(markdown, encoding="utf-8")
    archive_md.write_text(markdown, encoding="utf-8")
    return latest_md, archive_md


def main() -> int:
    """评测显式提供的分组权重。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="*")
    parser.add_argument("--model", help="只评测一个 yolo.<group> 模型")
    parser.add_argument(
        "--weights",
        action="append",
        default=[],
        help="可重复：group=/path/to/best.pt",
    )
    parser.add_argument("--version")
    args = parser.parse_args()
    groups = load_pipeline_groups()
    requested = [group_for_model_id(args.model, groups)] if args.model else args.groups
    selected = collect_groups(requested, groups)
    weights = _parse_weights(args.weights)
    missing = [group for group in selected if group not in weights]
    if missing:
        raise SystemExit(
            "迁移后的 benchmark 不再猜测历史 runs 权重；请提供 "
            + " ".join(f"--weights {group}=/path/to/best.pt" for group in missing)
        )
    results = [_run_group(group, weights[group]) for group in selected]
    latest_md, archive_md = write_summary(results, args.version)
    print(f"已写入 {latest_md}")
    print(f"已归档 {archive_md}")
    return 0 if all(item["exit_code"] == 0 for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
