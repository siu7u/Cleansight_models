"""Build temporal model test inputs from pinned feature and label sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_mapping(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        idx, label = line.split(maxsplit=1)
        mapping[label] = int(idx)
    return mapping


def load_sequence(
    features_dir: Path, labels_dir: Path, mapping: dict[str, int], name: str
) -> tuple[np.ndarray, np.ndarray]:
    features = np.load(features_dir / f"{name}.npy")
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D: {name} shape={features.shape}")
    if features.shape[0] < features.shape[1]:
        # Historical MS-TCN features are often [F, T]; temporal baselines use [T, F].
        features = features.T

    labels = []
    for line in (labels_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            labels.append(mapping[line])
    y = np.asarray(labels, dtype=np.int64)
    common_len = min(len(features), len(y))
    return features[:common_len].astype(np.float32), y[:common_len]


def build_stream_windows(x: np.ndarray, y: np.ndarray, window: int):
    """Yield causal windows where prediction at t sees only frames <= t."""
    for end in range(window, len(x) + 1):
        yield x[end - window : end], y[end - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/Endo_Project")
    parser.add_argument("--name", required=True)
    parser.add_argument("--window", type=int, default=64)
    args = parser.parse_args()

    root = Path(args.data_root)
    mapping = load_mapping(root / "mapping.txt")
    x, y = load_sequence(root / "features", root / "groundTruth", mapping, args.name)
    count = sum(1 for _ in build_stream_windows(x, y, args.window))
    print(
        {
            "name": args.name,
            "frames": len(x),
            "feature_dim": int(x.shape[1]),
            "labels": len(y),
            "window": args.window,
            "windows": count,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
