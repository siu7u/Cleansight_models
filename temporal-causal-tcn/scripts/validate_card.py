#!/usr/bin/env python3
"""Check CARD.md contains the required online gate fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.validate_card_gate import validate_card


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("card", nargs="?", default="CARD.md")
    args = parser.parse_args()

    path = Path(args.card)
    errors = validate_card(path)
    if errors:
        print(f"card validation failed: {path}")
        for error in errors:
            print(f"missing gate: {error}")
        return 1
    print(f"card validation passed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
