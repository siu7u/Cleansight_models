"""稳定交付 manifest：只描述文件、摘要和 schema，不负责复制、上传或发布。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


DELIVERY_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_schema_version(path: Path) -> int | None:
    """JSON 文件若声明 schema_version，则将其写入交付清单。"""

    if path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("schema_version") if isinstance(payload, Mapping) else None
    return int(value) if isinstance(value, int) else None


def build_delivery_manifest(
    *,
    run_id: str,
    model_id: str,
    base_dir: str | Path,
    files: Iterable[tuple[str, str | Path, bool]],
) -> dict[str, Any]:
    """构造 packaging-ready 文件清单；每项为 (role, path, required)。"""

    base = Path(base_dir).resolve()
    items = []
    for role, raw_path, required in files:
        path = Path(raw_path).resolve()
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"交付必需文件不存在: role={role}, path={path}")
            continue
        try:
            display = path.relative_to(base).as_posix()
            portable = True
        except ValueError:
            display = str(path)
            portable = False
        items.append(
            {
                "role": str(role),
                "path": display,
                "portable": portable,
                "required": bool(required),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "content_schema_version": _content_schema_version(path),
            }
        )
    manifest = {
        "schema_version": DELIVERY_SCHEMA_VERSION,
        "manifest_type": "cleansight_model_delivery",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model_id": model_id,
        "base_dir": str(base),
        "files": sorted(items, key=lambda item: (item["role"], item["path"])),
    }
    validate_delivery_manifest(manifest)
    return manifest


def validate_delivery_manifest(manifest: Mapping[str, Any]) -> None:
    """验证稳定交付结构，不访问文件系统也不执行发布决策。"""

    if manifest.get("schema_version") != DELIVERY_SCHEMA_VERSION:
        raise ValueError(f"不支持 delivery schema_version={manifest.get('schema_version')!r}")
    if manifest.get("manifest_type") != "cleansight_model_delivery":
        raise ValueError("delivery manifest_type 非法")
    for key in ("created_at", "run_id", "model_id", "base_dir"):
        if not manifest.get(key):
            raise ValueError(f"delivery manifest 缺少 {key}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("delivery manifest 缺少 files")
    roles = {item.get("role") for item in files if isinstance(item, Mapping)}
    for required_role in ("checkpoint", "evaluation"):
        if required_role not in roles:
            raise ValueError(f"delivery manifest 缺少必需 role={required_role}")
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("delivery files item 必须是映射")
        for key in ("role", "path", "sha256", "size_bytes", "required", "portable"):
            if key not in item:
                raise ValueError(f"delivery files item 缺少 {key}")
        if len(str(item["sha256"])) != 64:
            raise ValueError(f"delivery 文件摘要非法: {item['path']}")


def write_delivery_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """验证并确定性写出 UTF-8 交付清单。"""

    validate_delivery_manifest(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
