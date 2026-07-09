#!/usr/bin/env python3
"""Validate CARD.md online-release gate fields.

Any model promoted to online use must document three fields in CARD.md:

- measured runtime latency on the deployment machine
- receptive field and by-construction causality
- parameter count
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PENDING_VALUES = {
    "",
    "todo",
    "tbd",
    "待测",
    "未测",
    "待补",
    "未知",
    "na",
    "n/a",
    "none",
    "null",
}
NON_CAUSAL_TERMS = ("非因果", "否", "no", "false", "offline", "仅离线")


def extract_gate_section(text: str) -> str:
    """Return the `上线门禁` section text, or all text if the section is absent."""

    match = re.search(r"^##\s*上线门禁\s*$", text, flags=re.MULTILINE)
    if not match:
        return text
    rest = text[match.end() :]
    next_heading = re.search(r"^##\s+", rest, flags=re.MULTILINE)
    return rest[: next_heading.start()] if next_heading else rest


def field_value(section: str, names: tuple[str, ...]) -> str | None:
    """Find a Markdown list/table field value by any accepted Chinese name."""

    for name in names:
        bullet = re.search(rf"^\s*[-*]\s*{re.escape(name)}\s*[：:]\s*(.+?)\s*$", section, flags=re.MULTILINE)
        if bullet:
            return bullet.group(1).strip()
        table = re.search(rf"^\|\s*{re.escape(name)}\s*\|\s*(.+?)\s*\|", section, flags=re.MULTILINE)
        if table:
            return table.group(1).strip()
    return None


def is_filled(value: str | None) -> bool:
    """Return whether a CARD field is present and not a placeholder."""

    if value is None:
        return False
    clean = value.strip().strip("`*_ ").lower()
    return clean not in PENDING_VALUES


def validate_card(path: Path) -> list[str]:
    """Return validation errors for CARD online-release gate fields."""

    text = path.read_text(encoding="utf-8")
    section = extract_gate_section(text)
    errors: list[str] = []

    latency = field_value(section, ("运行延迟", "单 tick 延迟", "单tick延迟", "single_tick_latency"))
    receptive = field_value(section, ("感受域", "感受野", "receptive_field_frames"))
    causal = field_value(section, ("因果性", "causality", "online_causal"))
    params = field_value(section, ("模型参数量", "参数量", "params"))

    if not is_filled(latency):
        errors.append("missing or pending runtime latency: 运行延迟 / 单 tick 延迟")
    elif not re.search(r"\d", latency or ""):
        errors.append(f"runtime latency must contain a measured numeric value: {latency}")

    if not is_filled(receptive):
        errors.append("missing or pending receptive field: 感受域 / 感受野")

    if not is_filled(causal):
        errors.append("missing or pending causality: 因果性")
    elif any(term in causal.lower() for term in NON_CAUSAL_TERMS):
        errors.append(f"online causality must be by-construction causal, got: {causal}")

    if not is_filled(params):
        errors.append("missing or pending parameter count: 模型参数量 / 参数量")
    elif not re.search(r"\d", params or ""):
        errors.append(f"parameter count must contain a numeric value: {params}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CARD.md online-release gate fields.")
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
