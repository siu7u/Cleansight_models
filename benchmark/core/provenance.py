"""EvaluationResult v2 的运行、checkpoint 与 testset 溯源信息。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from framework.cleansight_eval.core.environment import now_iso


_REPO_ROOT = Path(__file__).resolve().parents[2]


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
    """记录 checkpoint 路径与内容哈希，文件大小由 delivery manifest 统一保存。"""

    path = Path(checkpoint).resolve()
    info: dict[str, Any] = {
        "path": _portable_path(path, run_dir),
        "sha256": sha256_file(path),
    }
    meta_path = path.with_name(path.name + ".meta.json")
    if meta_path.is_file():
        meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        info["meta"] = {
            "path": _portable_path(meta_path, run_dir),
            "sha256": sha256_file(meta_path),
            "schema_version": meta_payload.get("schema_version", 0),
            "checkpoint_bound": bool((meta_payload.get("checkpoint_binding") or {}).get("sha256")),
        }
        if isinstance(meta_payload.get("dataset"), dict):
            info["training_dataset"] = meta_payload["dataset"]
    return info


def build_run_info(
    checkpoint: str | Path,
    config_path: str | Path,
) -> tuple[dict[str, Any], Path | None]:
    """构造评估运行身份；环境与 Git 信息不写入评估结果。"""

    run_dir = find_run_dir(checkpoint)
    info: dict[str, Any] = {
        "id": run_dir.name if run_dir is not None else f"external-{Path(checkpoint).stem}",
        "created_at": now_iso(),
        "config": _portable_path(Path(config_path), run_dir),
    }
    return info, run_dir


def resolve_testset_info(cfg: dict) -> dict[str, Any]:
    """从 benchmark catalog 解析固定 testset；未登记配置降级为显式 ad-hoc 记录。"""

    evaluation = cfg.get("evaluation") or {}
    testset_id = evaluation.get("testset_id")
    data = cfg.get("data") or {}
    split = data.get("split_eval") or data.get("eval_split") or "unknown"
    dataset_ref = data.get("dataset_ref")
    if not testset_id and not dataset_ref:
        return {
            "id": f"ad-hoc:{data.get('name', 'dataset')}:{split}",
            "registered": False,
            "dataset_version": data.get("name") or str(data.get("root") or data.get("data_yaml")),
            "split": split,
            "fingerprint_sha256": None,
            "validation_errors": ["配置未声明 evaluation.testset_id，无法钉定 benchmark testset"],
        }

    from framework.cleansight_eval.core.catalog import (
        get_dataset_split,
        load_testsets,
        manifest_fingerprint,
        read_split_items,
        resolve_path,
        validate_catalog,
    )

    catalog = load_testsets()
    if dataset_ref:
        derived = get_dataset_split(str(dataset_ref), str(split), catalog)
        if testset_id and testset_id != derived.id:
            raise ValueError(
                f"evaluation.testset_id={testset_id!r} 与 dataset_ref/split="
                f"{dataset_ref!r}/{split!r} 推导结果 {derived.id!r} 不一致"
            )
        testset_id = derived.id
    if testset_id not in catalog:
        raise KeyError(f"evaluation.testset_id={testset_id!r} 未登记到 framework/testsets.yaml")
    spec = catalog[testset_id]
    errors = list(validate_catalog(catalog).get(testset_id, []))
    if str(split) != spec.split:
        errors.append(f"配置评估 split={split!r} 与 testset split={spec.split!r} 不一致")
    if dataset_ref and dataset_ref != spec.dataset:
        errors.append(
            f"配置 dataset_ref={dataset_ref!r} 与 testset dataset={spec.dataset!r} 不一致"
        )
    configured_root = data.get("root")
    if configured_root and spec.data_root:
        actual_root = Path(str(configured_root)).expanduser().resolve()
        registered_root = resolve_path(spec.data_root, spec.root)
        if actual_root != registered_root:
            errors.append(
                f"配置 data.root={actual_root} 与 testset data_root={registered_root} 不一致"
            )
    configured_manifest = data.get("data_yaml")
    if configured_manifest and spec.family == "yolo":
        actual_manifest = Path(str(configured_manifest)).expanduser().resolve()
        registered_manifest = resolve_path(spec.manifest, spec.root)
        if actual_manifest != registered_manifest:
            errors.append(
                f"配置 data_yaml={actual_manifest} 与 testset manifest={registered_manifest} 不一致"
            )
    configured_dim = (cfg.get("feature_schema") or {}).get("dim")
    if configured_dim is not None and spec.input_dim is not None and configured_dim != spec.input_dim:
        errors.append(
            f"配置 feature dim={configured_dim} 与 testset input_dim={spec.input_dim} 不一致"
        )
    configured_mapping = (cfg.get("feature_schema") or {}).get("version")
    if configured_mapping is not None and configured_mapping != spec.feature_mapping:
        errors.append(
            f"配置 feature mapping={configured_mapping!r} 与 testset "
            f"feature_mapping={spec.feature_mapping!r} 不一致"
        )
    model = cfg.get("model") or {}
    if spec.input_dim is not None and model.get("input_dim") is not None:
        if model["input_dim"] != spec.input_dim:
            errors.append(
                f"配置 model.input_dim={model['input_dim']} 与 testset input_dim={spec.input_dim} 不一致"
            )
    if spec.labels and model.get("num_classes") is not None:
        if model["num_classes"] != len(spec.labels):
            errors.append(
                f"配置 model.num_classes={model['num_classes']} 与 testset labels="
                f"{len(spec.labels)} 不一致"
            )
    return {
        "id": spec.id,
        "registered": True,
        "dataset": spec.dataset,
        "dataset_version": spec.dataset_version,
        "dataset_revision": spec.dataset_revision,
        "split": spec.split,
        "split_overlap_policy": spec.split_overlap_policy,
        "fingerprint_sha256": manifest_fingerprint(spec),
        "labels": list(spec.labels),
        "num_items": len(read_split_items(spec)),
        "validation_errors": errors,
    }
