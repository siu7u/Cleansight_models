#!/usr/bin/env python3
"""Run detailed eval and latency for the three temporal baseline models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "benchmark" / "single_model"

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


def run_json(cmd: list[str], cwd: Path) -> tuple[dict | None, str, int]:
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


def write_summary(results: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "temporal_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 时序单模型 Benchmark 汇总",
        "",
        "| 模型 | checkpoint | Idle Recall | Long Recall | Short Recall | 延迟 | 状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        recalls = (item.get("eval") or {}).get("per_class_recall", {})
        latency = item.get("latency") or {}
        latency_value = latency.get("mean_ms") or latency.get("avg_ms") or latency.get("single_tick_latency_ms")
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
    (OUT_DIR / "temporal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[m["name"] for m in DEFAULT_MODELS], help="只跑一个模型")
    args = parser.parse_args()

    selected = [m for m in DEFAULT_MODELS if not args.model or m["name"] == args.model]
    results = [benchmark_model(item) for item in selected]
    write_summary(results)
    print(f"已写入 {OUT_DIR / 'temporal_summary.md'}")
    return 0 if all(r["eval_exit_code"] == 0 and r["latency_exit_code"] == 0 for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

