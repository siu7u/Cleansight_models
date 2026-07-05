#!/usr/bin/env python3
"""Check CARD.md contains the required online gate fields."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_TERMS = [
    "single_tick_latency",
    "receptive_field_frames",
    "params",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("card", nargs="?", default="CARD.md")
    args = parser.parse_args()

    path = Path(args.card)
    text = path.read_text(encoding="utf-8")
    missing = [term for term in REQUIRED_TERMS if term not in text]
    if missing:
        print(f"card validation failed: {path}")
        for term in missing:
            print(f"missing: {term}")
        return 1
    print(f"card validation passed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
