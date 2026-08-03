#!/usr/bin/env python3
"""通过统一 benchmark CLI 批量评测历史时序 checkpoint。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "benchmark" / "single_model"
LATEST_DIR = OUT_DIR / "latest"
DEFAULT_MODELS = [
    {
        "name": "gru",
        "config": "framework/experiments/legacy-gru-v1.yaml",
        "checkpoint": "registry/temporal/gru-v1/gru-final-20260704-150629.pt",
    },
    {
        "name": "tcn",
        "config": "framework/experiments/legacy-causal-tcn-v1.yaml",
        "checkpoint": "registry/temporal/causal-tcn-v1/tcn-final-20260704-160652.pt",
    },
    {
        "name": "transformer",
        "config": "framework/experiments/legacy-causal-transformer-v1.yaml",
        "checkpoint": (
            "registry/temporal/causal-transformer-v1/"
            "transformer-final-20260704-161653.pt"
        ),
    },
]


def build_run_id(version: str | None) -> str:
    """生成稳定归档编号。"""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not version:
        return timestamp
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", version.strip()).strip("-")
    return f"{slug or 'version'}-{timestamp}"


def benchmark_model(item: dict) -> dict:
    """调用统一 eval CLI；模型加载和推理只发生在 framework Pipeline。"""

    invocation_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir = LATEST_DIR / item["name"] / invocation_id
    command = [
        sys.executable,
        "-m",
        "benchmark.cli.eval",
        "--config",
        str(ROOT / item["config"]),
        "--ckpt",
        str(ROOT / item["checkpoint"]),
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
        **item,
        "evaluation": evaluation,
        "evaluation_path": (
            str(evaluation_path.relative_to(ROOT)) if evaluation_path is not None else None
        ),
        "exit_code": proc.returncode,
        "log_tail": (proc.stdout + proc.stderr)[-3000:],
    }


def _metric(item: dict, name: str) -> str:
    """从 EvaluationResult v2 读取一个三态指标。"""

    metric = (
        ((item.get("evaluation") or {}).get("metrics") or {}).get("summary") or {}
    ).get(name, {})
    if metric.get("state") != "computed":
        return metric.get("state", "MISSING")
    value = metric.get("value")
    return f"{float(value):.2f}" if isinstance(value, (int, float)) else str(value)


def write_summary(results: list[dict], version: str | None) -> tuple[Path, Path]:
    """汇总统一 EvaluationResult，不再解析历史 stdout。"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    archive_dir = OUT_DIR / "reports"
    archive_dir.mkdir(parents=True, exist_ok=True)
    run_id = build_run_id(version)
    payload = {
        "benchmark": "single_model_temporal",
        "version": version,
        "run_id": run_id,
        "models": results,
    }
    latest_json = LATEST_DIR / "temporal_summary.json"
    archive_json = archive_dir / f"temporal_summary_{run_id}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    latest_json.write_text(text, encoding="utf-8")
    archive_json.write_text(text, encoding="utf-8")

    lines = [
        "# 时序单模型 Benchmark 汇总",
        "",
        f"- 版本：`{version or run_id}`",
        "- 模型训练与推理：`framework`",
        "- 指标与结果：`benchmark`",
        "",
        "| 模型 | Accuracy | Edit | F1@0.5 | 结果 | 状态 |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in results:
        lines.append(
            f"| {item['name']} | {_metric(item, 'acc')} | {_metric(item, 'edit')} | "
            f"{_metric(item, 'f1@0.5')} | `{item.get('evaluation_path') or '未生成'}` | "
            f"{'OK' if item['exit_code'] == 0 else 'CHECK'} |"
        )
    latest_md = LATEST_DIR / "temporal_summary.md"
    archive_md = archive_dir / f"temporal_summary_{run_id}.md"
    markdown = "\n".join(lines) + "\n"
    latest_md.write_text(markdown, encoding="utf-8")
    archive_md.write_text(markdown, encoding="utf-8")
    return latest_md, archive_md


def main() -> int:
    """运行所选历史模型并写统一摘要。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[item["name"] for item in DEFAULT_MODELS])
    parser.add_argument("--version")
    args = parser.parse_args()
    selected = [
        item for item in DEFAULT_MODELS if not args.model or item["name"] == args.model
    ]
    results = [benchmark_model(item) for item in selected]
    latest_md, archive_md = write_summary(results, args.version)
    print(f"已写入 {latest_md}")
    print(f"已归档 {archive_md}")
    return 0 if all(item["exit_code"] == 0 for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
