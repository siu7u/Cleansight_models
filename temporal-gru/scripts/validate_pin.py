#!/usr/bin/env python3
"""Lightweight pin.yaml validator for temporal model repositories."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_KEYS = [
    "model:",
    "  name:",
    "  version:",
    "dataset:",
    "  view_b:",
    "detection:",
    "  yolo_version:",
    "feature_mapping:",
    "  version:",
    "  file:",
    "  sha256:",
    "runtime:",
    "  online_causal:",
    "  receptive_field_frames:",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pin", nargs="?", default="pin.yaml")
    args = parser.parse_args()

    path = Path(args.pin)
    text = path.read_text(encoding="utf-8")
    missing = [key for key in REQUIRED_KEYS if key not in text]
    if missing:
        print(f"pin validation failed: {path}")
        for key in missing:
            print(f"missing: {key}")
        return 1
    print(f"pin validation passed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
