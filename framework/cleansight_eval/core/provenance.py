"""schema v2 评估结果的运行、checkpoint 与 testset 溯源信息。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .environment import now_iso


_REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA-256，避免一次读入大型 checkpoint。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_run_dir(checkpoint: str | Path) -> Path | None:
    """从 ``<run>/checkpoints/...`` checkpoint 向上定位 run 目录。"""

    checkpoint = Path(checkpoint)
    for parent in checkpoint.parents:
        if parent.name == "checkpoints":
            return parent.parent
    return None


def _portable_path(path: Path, base: Path | None = None) -> str:
    """优先保存 run 相对路径，其次仓库相对路径，最后才保留绝对路径。"""

    path = path.resolve()
    if base is not None:
        try:
            return path.relative_to(base.resolve()).as_posix()
        except ValueError:
            pass
    try:
        return path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        pass
    return str(path)


def build_checkpoint_info(checkpoint: str | Path, run_dir: Path | None = None) -> dict[str, Any]:
    """记录 checkpoint 路径、大小与内容哈希，供版本发布核验。"""

    path = Path(checkpoint).resolve()
    info: dict[str, Any] = {
        "path": _portable_path(path, run_dir),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    meta_path = path.with_name(path.name + ".meta.json")
    if meta_path.is_file():
        info["meta"] = {
            "path": _portable_path(meta_path, run_dir),
            "sha256": sha256_file(meta_path),
        }
    return info


def build_run_info(
    checkpoint: str | Path,
    config_path: str | Path,
    device,
) -> tuple[dict[str, Any], Path | None]:
    """构造评估运行引用，关联训练 run 中的解析配置和环境快照。"""

    run_dir = find_run_dir(checkpoint)
    info: dict[str, Any] = {
        "id": run_dir.name if run_dir is not None else f"external-{Path(checkpoint).stem}",
        "created_at": now_iso(),
        "device": str(device),
        "config": _portable_path(Path(config_path), run_dir),
    }
    if run_dir is not None:
        resolved = run_dir / "config.resolved.json"
        environment = run_dir / "env.json"
        if resolved.is_file():
            info["resolved_config"] = _portable_path(resolved, run_dir)
        if environment.is_file():
            info["environment"] = _portable_path(environment, run_dir)
    return info, run_dir


def resolve_testset_info(cfg: dict) -> dict[str, Any]:
    """从 benchmark catalog 解析固定 testset；未登记配置降级为显式 ad-hoc 记录。"""

    evaluation = cfg.get("evaluation") or {}
    testset_id = evaluation.get("testset_id")
    data = cfg.get("data") or {}
    split = data.get("split_eval") or data.get("eval_split") or "unknown"
    if not testset_id:
        return {
            "id": f"ad-hoc:{data.get('name', 'dataset')}:{split}",
            "registered": False,
            "dataset_version": data.get("name") or str(data.get("root") or data.get("data_yaml")),
            "split": split,
            "purpose": "ad_hoc_evaluation",
            "manifest_sha256": None,
            "fingerprint_sha256": None,
            "validation_errors": ["配置未声明 evaluation.testset_id，无法钉定 benchmark testset"],
        }

    # benchmark 尚未安装成独立包，兼容从仓库根目录或 framework 目录执行。
    try:
        from benchmark.core.testsets import (
            load_testsets,
            manifest_fingerprint,
            read_split_items,
            resolve_path,
            validate_catalog,
        )
    except ModuleNotFoundError:  # pragma: no cover - 与 temporal.metrics 相同的 cwd 兼容路径
        import sys

        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from benchmark.core.testsets import (
            load_testsets,
            manifest_fingerprint,
            read_split_items,
            resolve_path,
            validate_catalog,
        )

    catalog = load_testsets()
    if testset_id not in catalog:
        raise KeyError(f"evaluation.testset_id={testset_id!r} 未登记到 benchmark/testsets.yaml")
    spec = catalog[testset_id]
    manifest = resolve_path(spec.manifest, spec.root)
    errors = list(validate_catalog(catalog).get(testset_id, []))
    if str(split) != spec.split:
        errors.append(f"配置评估 split={split!r} 与 testset split={spec.split!r} 不一致")
    configured_dim = (cfg.get("feature_schema") or {}).get("dim")
    if configured_dim is not None and spec.input_dim is not None and configured_dim != spec.input_dim:
        errors.append(
            f"配置 feature dim={configured_dim} 与 testset input_dim={spec.input_dim} 不一致"
        )
    return {
        "id": spec.id,
        "registered": True,
        "family": spec.family,
        "dataset_version": spec.dataset_version,
        "split": spec.split,
        "purpose": spec.purpose,
        "manifest": _portable_path(manifest),
        "manifest_sha256": sha256_file(manifest),
        "fingerprint_sha256": manifest_fingerprint(spec),
        "feature_mapping": spec.feature_mapping,
        "input_dim": spec.input_dim,
        "labels": list(spec.labels),
        "num_items": len(read_split_items(spec)),
        "validation_errors": errors,
    }
