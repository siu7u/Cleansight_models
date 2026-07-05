#!/usr/bin/env python3
"""Measure single-window temporal model forward latency."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


def build_model(model_name: str, input_dim: int, num_classes: int):
    if model_name == "gru":
        from model import GRUClassifier

        return GRUClassifier(input_dim, num_classes)
    if model_name == "tcn":
        from model import TCNClassifier

        return TCNClassifier(input_dim, num_classes)
    if model_name == "transformer":
        from model import TransformerClassifier

        return TransformerClassifier(input_dim, num_classes)
    raise ValueError(f"unknown model: {model_name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True, choices=["gru", "tcn", "transformer"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.input_dim, args.num_classes).to(device)
    model.load_state_dict(torch.load(repo / args.checkpoint, map_location=device))
    model.eval()
    x = torch.randn(1, args.window, args.input_dim, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(args.warmup):
            model(x)
        sync()
        samples = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            model(x)
            sync()
            samples.append((time.perf_counter() - start) * 1000.0)

    result = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "window": args.window,
        "input_dim": args.input_dim,
        "repeat": args.repeat,
        "mean_ms": round(statistics.mean(samples), 4),
        "median_ms": round(statistics.median(samples), 4),
        "p95_ms": round(sorted(samples)[int(0.95 * (len(samples) - 1))], 4),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
