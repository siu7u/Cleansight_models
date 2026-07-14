#!/usr/bin/env python3
"""运行三个时序基线模型的详细评测和延迟评测。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "benchmark" / "single_model"
LATEST_DIR = OUT_DIR / "latest"

DEFAULT_MODELS = [
    {
        "name": "gru",
        "repo": "temporal-gru",
        "checkpoint": "registry/gru-v1/gru-final-20260704-150629.pt",
        "input_dim": 20,
        "window": 64,
    },
    {
        "name": "tcn",
        "repo": "temporal-causal-tcn",
        "checkpoint": "registry/tcn-v1/tcn-final-20260704-160652.pt",
        "input_dim": 20,
        "window": 64,
    },
    {
        "name": "transformer",
        "repo": "temporal-transformer",
        "checkpoint": "registry/transformer-v1/transformer-final-20260704-161653.pt",
        "input_dim": 20,
        "window": 64,
    },
]


def build_run_id(version: str | None) -> str:
    """生成用于归档 benchmark summary 的版本化运行编号。"""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not version:
        return timestamp
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", version.strip()).strip("-")
    return f"{slug or 'version'}-{timestamp}"


def run_json(cmd: list[str], cwd: Path) -> tuple[dict | None, str, int]:
    """运行 benchmark 子进程，并从 stdout 中解析第一个 JSON 对象。

    返回可解析的 JSON、用于排查问题的合并日志，以及进程退出码。子进程可能会
    加载 checkpoint 或使用 CUDA。
    """

    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    text = proc.stdout.strip()
    data = None
    if text:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                data = None
    return data, proc.stdout + proc.stderr, proc.returncode


def benchmark_model(item: dict) -> dict:
    """对一个时序模型运行详细评测和延迟测量。"""

    repo = ROOT / item["repo"]
    eval_cmd = [
        sys.executable,
        str(ROOT / "tools" / "eval_temporal_detailed.py"),
        "--repo",
        ".",
        "--model",
        item["name"],
        "--checkpoint",
        item["checkpoint"],
    ]
    latency_cmd = [
        sys.executable,
        str(ROOT / "tools" / "measure_temporal_latency.py"),
        "--repo",
        ".",
        "--model",
        item["name"],
        "--checkpoint",
        item["checkpoint"],
        "--window",
        str(item["window"]),
        "--input-dim",
        str(item["input_dim"]),
    ]

    eval_data, eval_log, eval_code = run_json(eval_cmd, repo)
    latency_data, latency_log, latency_code = run_json(latency_cmd, repo)
    return {
        **item,
        "eval": eval_data,
        "latency": latency_data,
        "eval_exit_code": eval_code,
        "latency_exit_code": latency_code,
        "eval_log_tail": eval_log[-2000:],
        "latency_log_tail": latency_log[-2000:],
    }


def write_summary(results: list[dict], version: str | None) -> tuple[Path, Path]:
    """将时序单模型 benchmark 汇总写成 JSON 和 Markdown。"""

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
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    latest_json = LATEST_DIR / "temporal_summary.json"
    archive_json = archive_dir / f"temporal_summary_{run_id}.json"
    latest_json.write_text(json_text, encoding="utf-8")
    archive_json.write_text(json_text, encoding="utf-8")

    lines = [
        "# 时序单模型 Benchmark 汇总",
        "",
        f"- 版本：`{version or run_id}`",
        f"- 归档编号：`{run_id}`",
        "",
        "| 模型 | checkpoint | Idle Recall | Long Recall | Short Recall | Model-forward 延迟 | 状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        recalls = (item.get("eval") or {}).get("per_class_recall", {})
        latency = item.get("latency") or {}
        latency_value = latency.get("model_forward_mean_ms") or latency.get("mean_ms") or latency.get("avg_ms")
        status = "OK" if item["eval_exit_code"] == 0 and item["latency_exit_code"] == 0 else "CHECK"
        lines.append(
            "| {name} | `{ckpt}` | {idle} | {long} | {short} | {latency} | {status} |".format(
                name=item["name"],
                ckpt=item["checkpoint"],
                idle=f"{recalls.get('Idle', 0) * 100:.2f}%" if "Idle" in recalls else "待测",
                long=f"{recalls.get('Long_Brushing', 0) * 100:.2f}%" if "Long_Brushing" in recalls else "待测",
                short=f"{recalls.get('Short_Brushing', 0) * 100:.2f}%" if "Short_Brushing" in recalls else "待测",
                latency=f"{float(latency_value):.3f} ms" if latency_value is not None else "待测",
                status=status,
            )
        )
    md_text = "\n".join(lines) + "\n"
    latest_md = LATEST_DIR / "temporal_summary.md"
    archive_md = archive_dir / f"temporal_summary_{run_id}.md"
    latest_md.write_text(md_text, encoding="utf-8")
    archive_md.write_text(md_text, encoding="utf-8")
    return latest_md, archive_md


def main() -> int:
    """解析命令行参数，运行选定的时序 benchmark，并返回整体状态。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[m["name"] for m in DEFAULT_MODELS], help="只跑一个模型")
    parser.add_argument("--version", help="为本次 benchmark summary 指定版本名，例如 temporal-v2")
    args = parser.parse_args()

    selected = [m for m in DEFAULT_MODELS if not args.model or m["name"] == args.model]
    results = [benchmark_model(item) for item in selected]
    latest_md, archive_md = write_summary(results, args.version)
    print(f"已写入 {latest_md}")
    print(f"已归档 {archive_md}")
    return 0 if all(r["eval_exit_code"] == 0 and r["latency_exit_code"] == 0 for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
