"""所有单模型、时序喂法与端到端评估共用的 JSON envelope。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
ALLOWED_STATUS = {"PASS", "FAIL", "PENDING", "EXPLORATORY"}


def make_run_id(prefix: str) -> str:
    """生成可用于文件名与 CARD marker 的 UTC 运行编号。"""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", prefix).strip("-") or "evaluation"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{slug}-{timestamp}"


def build_result(
    *,
    benchmark: str,
    task_type: str,
    run_id: str,
    model: dict[str, Any] | None,
    testset: dict[str, Any],
    inference: dict[str, Any],
    metrics: dict[str, Any],
    status: str,
    reasons: list[str] | None = None,
    limits: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造并校验统一评估结果；所有比率由 metric_spec 声明为 0..1。"""

    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "task_type": task_type,
        "run": {
            "id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "model": model,
        "testset": testset,
        "inference": inference,
        "metrics": metrics,
        "limits": limits or {"is_smoke": False},
        "gates": {"status": status, "reasons": reasons or []},
        "artifacts": artifacts or {},
    }
    validate_result(result)
    return result


def validate_result(result: dict[str, Any]) -> None:
    """检查公共 envelope 的必填字段和门禁状态。"""

    required = {
        "schema_version",
        "benchmark",
        "task_type",
        "run",
        "model",
        "testset",
        "inference",
        "metrics",
        "limits",
        "gates",
        "artifacts",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"评估结果缺少字段: {missing}")
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"不支持 schema_version={result['schema_version']}")
    status = result.get("gates", {}).get("status")
    if status not in ALLOWED_STATUS:
        raise ValueError(f"非法 gates.status={status}")
    testset = result.get("testset") or {}
    for key in ("id", "dataset_version", "split", "manifest_sha256"):
        if not testset.get(key):
            raise ValueError(f"testset 缺少 {key}")


def write_result(path: Path, result: dict[str, Any]) -> Path:
    """校验后写出 UTF-8 JSON，供 CARD、release gate 和归档共用。"""

    validate_result(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
