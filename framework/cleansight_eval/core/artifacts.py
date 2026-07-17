"""framework 评估 artifact 的确定性落盘与哈希引用。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .provenance import sha256_file


def _verify_recomputable(payload: dict[str, Any]) -> bool | None:
    """调用 benchmark 唯一校验器；检测需结合 testset 真值，返回 None。"""

    try:
        from benchmark.core.artifacts import prediction_artifact_recomputable
    except ModuleNotFoundError:  # pragma: no cover - 从 framework 目录执行时触发
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from benchmark.core.artifacts import prediction_artifact_recomputable
    return prediction_artifact_recomputable(payload)


def write_json_artifact(
    path: str | Path,
    payload: dict[str, Any],
    *,
    relative_to: str | Path | None = None,
) -> dict[str, Any]:
    """写 JSON artifact，并返回可写入 EvaluationResult 的路径、schema 和 SHA-256。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    display = path.resolve()
    if relative_to is not None:
        try:
            display = display.relative_to(Path(relative_to).resolve())
        except ValueError:
            pass
    reference = {
        "path": display.as_posix(),
        "sha256": sha256_file(path),
        "schema_version": payload.get("schema_version"),
    }
    recomputable = _verify_recomputable(payload)
    if recomputable is not None:
        reference["recomputable"] = recomputable
    return reference
