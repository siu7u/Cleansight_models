"""异构评估矩阵汇总（framework 层）。

需求 §9：矩阵同时支持人读与机读，允许不同模型拥有不同指标列，不适用显示
``N/A``、缺失显示 ``MISSING``、成功计算显示数值，三者严格区分；不对异构指标
生成统一综合分数。
"""

from __future__ import annotations

import json
from pathlib import Path

from .envelope import EvalEnvelope, MetricState


def collect_envelopes(runs_dir: str | Path) -> list[EvalEnvelope]:
    """递归扫描 runs 目录下所有 ``*.envelope.json``。"""

    runs_dir = Path(runs_dir)
    return [EvalEnvelope.read(p) for p in sorted(runs_dir.rglob("*.envelope.json"))]


def _metric_columns(envelopes: list[EvalEnvelope]) -> list[str]:
    cols: list[str] = []
    for env in envelopes:
        for name in list(env.metrics) + [f"perf.{k}" for k in env.performance]:
            if name not in cols:
                cols.append(name)
    return cols


def build_matrix(envelopes: list[EvalEnvelope]) -> dict:
    """构建机读矩阵：固定标识列 + 异构指标列，保留三态。"""

    columns = _metric_columns(envelopes)
    rows = []
    for env in envelopes:
        cells: dict[str, dict] = {}
        for name, mv in env.metrics.items():
            cells[name] = mv.to_dict()
        for name, mv in env.performance.items():
            cells[f"perf.{name}"] = mv.to_dict()
        rows.append(
            {
                "family": env.family,
                "model_id": env.model_id,
                "task": env.task,
                "feeding": env.feeding,
                "checkpoint": env.checkpoint,
                "dataset": env.dataset,
                "feature_schema": env.feature_schema,
                "num_params": env.num_params,
                "integrity_ok": env.integrity.get("ok"),
                "cells": cells,
            }
        )
    return {"metric_columns": columns, "rows": rows}


def _cell_display(cell: dict | None) -> str:
    if cell is None:
        return ""  # 该模型没有这一列 → 空白，非 N/A、非 MISSING
    state = MetricState(cell["state"])
    if state is MetricState.NOT_APPLICABLE:
        return "N/A"
    if state is MetricState.MISSING:
        return "MISSING"
    return str(cell.get("value"))


def render_markdown(matrix: dict) -> str:
    """渲染人读 Markdown 表格。"""

    cols = matrix["metric_columns"]
    header = ["family", "model_id", "feeding", "params", "integrity"] + cols
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in matrix["rows"]:
        base = [
            row["family"],
            row["model_id"],
            row["feeding"],
            str(row.get("num_params") if row.get("num_params") is not None else ""),
            "ok" if row.get("integrity_ok") else "check",
        ]
        cells = [_cell_display(row["cells"].get(c)) for c in cols]
        lines.append("| " + " | ".join(base + cells) + " |")
    note = (
        "\n> 图例：`N/A`=指标不适用；`MISSING`=适用但缺失/失败；空白=该模型无此列。"
        "不对异构指标生成综合分数。\n"
    )
    return "# 评估矩阵\n\n" + "\n".join(lines) + "\n" + note


def write_matrix(runs_dir: str | Path, out_dir: str | Path | None = None) -> tuple[Path, Path]:
    """汇总并写出 ``matrix.json`` 与 ``matrix.md``。"""

    runs_dir = Path(runs_dir)
    out_dir = Path(out_dir) if out_dir else runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    envelopes = collect_envelopes(runs_dir)
    matrix = build_matrix(envelopes)

    json_path = out_dir / "matrix.json"
    md_path = out_dir / "matrix.md"
    json_path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(matrix), encoding="utf-8")
    return json_path, md_path
