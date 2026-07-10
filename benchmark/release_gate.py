#!/usr/bin/env python3
"""把 benchmark 汇总和上线元数据合并成一个 PASS/FAIL 门禁。"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmark" / "release_gate"
LATEST_DIR = OUT_DIR / "latest"
DEFAULT_SUMMARIES = [
    ROOT / "benchmark" / "single_model" / "latest" / "yolo_summary.json",
    ROOT / "benchmark" / "single_model" / "latest" / "temporal_summary.json",
    ROOT / "benchmark" / "temporal_feed_mode" / "latest" / "feed_mode_summary.json",
]


def build_run_id(version: str | None) -> str:
    """生成文件名安全的 release gate 运行编号。"""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not version:
        return timestamp
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", version.strip()).strip("-")
    return f"{slug or 'version'}-{timestamp}"


def load_json(path: Path) -> dict[str, Any]:
    """读取一份 benchmark JSON 汇总。"""

    return json.loads(path.read_text(encoding="utf-8"))


def status_of_summary(data: dict[str, Any]) -> str:
    """从 benchmark 汇总中解析 PASS/FAIL/UNKNOWN 状态。"""

    status = data.get("status")
    if status:
        return str(status)
    groups = data.get("groups")
    if isinstance(groups, list) and groups:
        return "PASS" if all(item.get("status") == "PASS" for item in groups) else "FAIL"
    models = data.get("models")
    if isinstance(models, list) and models:
        bad = [item for item in models if item.get("eval_exit_code", 0) or item.get("latency_exit_code", 0)]
        return "FAIL" if bad else "PASS"
    return "UNKNOWN"


def card_has(card_text: str, patterns: list[str]) -> bool:
    """检查 CARD 文本是否包含指定证据关键词。"""

    return any(pattern in card_text for pattern in patterns)


def collect_card_evidence(card: Path | None) -> dict[str, bool]:
    """检查 CARD 中是否能找到上线三项必填的证据。"""

    if not card or not card.exists():
        return {"latency": False, "causality": False, "num_params": False}
    text = card.read_text(encoding="utf-8")
    return {
        "latency": card_has(text, ["运行延迟", "部署机实测", "latency", "p95_ms", "mean_ms"]),
        "causality": card_has(text, ["感受域", "因果", "causal", "receptive"]),
        "num_params": card_has(text, ["参数量", "num_params", "parameters"]),
    }


def parse_args() -> argparse.Namespace:
    """解析 release gate 命令行输入。"""

    parser = argparse.ArgumentParser(description="CleanSight release gate")
    parser.add_argument("--version", help="本次 release gate 的版本名")
    parser.add_argument(
        "--summary",
        action="append",
        help="benchmark summary JSON 路径；可传多次。不传则读取已存在的 latest summary。",
    )
    parser.add_argument("--card", help="模型 CARD.md 路径，用于检查上线三项必填")
    parser.add_argument("--latency-ms", type=float, help="部署机实测运行延迟，毫秒")
    parser.add_argument(
        "--causality",
        choices=("causal", "by-construction-causal", "offline-only", "non-causal"),
        help="模型在线因果性/感受域声明",
    )
    parser.add_argument("--num-params", type=int, help="模型参数量")
    return parser.parse_args()


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    """根据 benchmark 汇总和必填元数据生成门禁结果。"""

    summary_paths = [Path(item) for item in args.summary] if args.summary else [p for p in DEFAULT_SUMMARIES if p.exists()]
    summaries = []
    reasons = []
    for path in summary_paths:
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            reasons.append(f"缺少 benchmark summary: {path}")
            continue
        data = load_json(path)
        status = status_of_summary(data)
        summaries.append(
            {
                "path": str(path.relative_to(ROOT)),
                "benchmark": data.get("benchmark"),
                "version": data.get("version"),
                "status": status,
            }
        )
        if status != "PASS":
            reasons.append(f"{path.relative_to(ROOT)} status={status}")

    if not summaries:
        reasons.append("缺少可读取的 benchmark summary")

    card_path = Path(args.card) if args.card else None
    if card_path and not card_path.is_absolute():
        card_path = ROOT / card_path
    card_evidence = collect_card_evidence(card_path)
    required = {
        "latency": args.latency_ms is not None or card_evidence["latency"],
        "causality": args.causality is not None or card_evidence["causality"],
        "num_params": args.num_params is not None or card_evidence["num_params"],
    }
    if not required["latency"]:
        reasons.append("缺少运行延迟(部署机实测)")
    if not required["causality"]:
        reasons.append("缺少感受域/因果性声明")
    if not required["num_params"]:
        reasons.append("缺少模型参数量")

    if args.causality in {"non-causal", "offline-only"}:
        reasons.append(f"在线 release 不允许 causality={args.causality}")

    return {
        "schema_version": 1,
        "benchmark": "release_gate",
        "version": args.version,
        "run_id": build_run_id(args.version),
        "summaries": summaries,
        "required_release_fields": {
            "latency_ms": args.latency_ms,
            "causality": args.causality,
            "num_params": args.num_params,
            "card": str(card_path.relative_to(ROOT)) if card_path and card_path.exists() else None,
            "present": required,
        },
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
    }


def write_outputs(result: dict[str, Any]) -> tuple[Path, Path]:
    """写入 latest 和归档版 release gate JSON/Markdown 报告。"""

    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    archive_dir = OUT_DIR / "reports"
    archive_dir.mkdir(parents=True, exist_ok=True)
    run_id = result["run_id"]

    json_text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    latest_json = LATEST_DIR / "release_gate.json"
    archive_json = archive_dir / f"release_gate_{run_id}.json"
    latest_json.write_text(json_text, encoding="utf-8")
    archive_json.write_text(json_text, encoding="utf-8")

    lines = [
        "# Release Gate",
        "",
        f"- 版本: `{result.get('version') or run_id}`",
        f"- 归档编号: `{run_id}`",
        f"- 结论: **{result['status']}**",
        "",
        "## Benchmark Summaries",
        "",
        "| Benchmark | Version | Status | Path |",
        "| --- | --- | --- | --- |",
    ]
    for item in result["summaries"]:
        lines.append(f"| {item.get('benchmark')} | {item.get('version')} | {item['status']} | `{item['path']}` |")

    fields = result["required_release_fields"]
    present = fields["present"]
    lines += [
        "",
        "## Required Release Fields",
        "",
        "| Field | Present | Value |",
        "| --- | --- | --- |",
        f"| 运行延迟 | {present['latency']} | `{fields.get('latency_ms')}` |",
        f"| 感受域/因果性 | {present['causality']} | `{fields.get('causality')}` |",
        f"| 模型参数量 | {present['num_params']} | `{fields.get('num_params')}` |",
    ]
    if result["reasons"]:
        lines += ["", "## Fail Reasons", ""]
        lines += [f"- {reason}" for reason in result["reasons"]]
    else:
        lines += ["", "全部门禁通过。"]

    md_text = "\n".join(lines) + "\n"
    latest_md = LATEST_DIR / "release_gate.md"
    archive_md = archive_dir / f"release_gate_{run_id}.md"
    latest_md.write_text(md_text, encoding="utf-8")
    archive_md.write_text(md_text, encoding="utf-8")
    return latest_md, archive_md


def main() -> int:
    """运行 release gate。"""

    args = parse_args()
    result = build_result(args)
    latest_md, archive_md = write_outputs(result)
    print(f"已写入 {latest_md}")
    print(f"已归档 {archive_md}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
