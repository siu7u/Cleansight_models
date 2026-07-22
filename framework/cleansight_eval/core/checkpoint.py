"""checkpoint 保存/加载 + 重建元信息（framework 层）。

需求 §7.1/§7.2/§8.1：checkpoint 必须携带能正确重建模型的信息，并且能确认
自身对应的模型配置；评估时不能因错误配置而静默加载。

实现方式：权重存 ``<path>``，重建元信息存 sidecar ``<path>.meta.json``。
元信息包含 type、模型配置、feature schema、input_dim/num_classes/window、pipeline，
时序训练还会写入数据集 version/revision、split fingerprint 与类别映射摘要。
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import torch

from .integrity import assert_checkpoint_config

META_SUFFIX = ".meta.json"
META_SCHEMA_VERSION = 1


def meta_path_for(checkpoint_path: str | Path) -> Path:
    return Path(str(checkpoint_path) + META_SUFFIX)


def _sha256_file(path: str | Path) -> str:
    """流式计算 checkpoint 摘要，避免大型权重一次读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_state_dict(payload):
    """兼容本框架 ``model_state``、常见 ``state_dict`` 包装和裸 state dict。"""

    if isinstance(payload, dict):
        for key in ("model_state", "state_dict"):
            state = payload.get(key)
            if isinstance(state, dict):
                return state
    return payload


def write_meta(checkpoint_path: str | Path, meta: dict) -> Path:
    """写 schema v1 sidecar，并与当前 checkpoint 内容和大小绑定。"""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在，无法写 metadata: {checkpoint}")
    payload = dict(meta)
    payload["schema_version"] = META_SCHEMA_VERSION
    payload["checkpoint_binding"] = {
        "sha256": _sha256_file(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
    }
    path = meta_path_for(checkpoint)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def save_checkpoint(path: str | Path, state_dict: dict, meta: dict) -> Path:
    """保存权重与重建元信息。

    ``meta`` 至少应包含 type、input_dim、num_classes；建议补充 model 配置、
    feature_schema、window、pipeline 和 dataset 溯源，以便评估阶段无需人工重复填写即可
    重建模型，并在 resume 时拒绝训练数据静默漂移。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    write_meta(path, meta)
    return path


def save_training_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    epoch: int,
    meta: dict,
    best_metric: dict | None = None,
    scheduler=None,
) -> Path:
    """保存可恢复训练的完整状态；同时复用原有 meta sidecar。"""

    payload = {
        "schema_version": 1,
        "checkpoint_kind": "training_state",
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": best_metric or {},
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    return save_checkpoint(path, payload, meta)


def load_meta(path: str | Path, *, require_schema: bool = False) -> dict:
    """读取并校验 sidecar；schema v1 必须与 checkpoint 内容绑定。

    ``require_schema=False`` 只用于迁移期读取历史 sidecar；正式评估应传 True。
    """

    mp = meta_path_for(path)
    if not mp.exists():
        raise FileNotFoundError(
            f"未找到 checkpoint 元信息 {mp}；无法确认该权重对应的模型配置，拒绝加载。"
        )
    meta = json.loads(mp.read_text(encoding="utf-8"))
    version = meta.get("schema_version")
    if version is None:
        if require_schema:
            raise ValueError(f"{mp} 是无 schema 的历史 metadata，正式评估拒绝加载")
        meta["_metadata_integrity"] = {"schema_version": 0, "bound": False}
        return meta
    if version != META_SCHEMA_VERSION:
        raise ValueError(f"不支持 metadata schema_version={version!r}")
    binding = meta.get("checkpoint_binding") or {}
    if not binding.get("sha256") or binding.get("size_bytes") is None:
        raise ValueError(f"{mp} 缺少 checkpoint_binding")
    checkpoint = Path(path)
    actual_size = checkpoint.stat().st_size
    actual_sha256 = _sha256_file(checkpoint)
    if int(binding["size_bytes"]) != actual_size or binding["sha256"] != actual_sha256:
        raise ValueError(f"{checkpoint} 内容与 metadata 绑定摘要不一致，拒绝加载")
    meta["_metadata_integrity"] = {"schema_version": version, "bound": True}
    return meta


def load_checkpoint(
    path: str | Path,
    expected: dict | None = None,
    map_location="cpu",
    *,
    require_meta_schema: bool = False,
    fallback_meta: dict | None = None,
):
    """加载权重并校验配置兼容性。

    返回 ``(state_dict, meta)``。若 ``expected`` 与元信息不兼容，立即抛
    ``CompatibilityError``，实现"不因错误配置静默加载"。``fallback_meta`` 只供调用方
    在明确的 exploratory 外部权重流程中使用；formal 模式禁止用配置声明代替绑定 sidecar。
    """

    if require_meta_schema and fallback_meta is not None:
        raise ValueError("formal checkpoint 加载不能使用 fallback metadata")
    try:
        meta = load_meta(path, require_schema=require_meta_schema)
    except FileNotFoundError:
        if fallback_meta is None:
            raise
        meta = dict(fallback_meta)
        meta["_metadata_integrity"] = {
            "schema_version": 0,
            "bound": False,
            "source": "config_fallback",
        }
    assert_checkpoint_config(meta, expected)
    payload = torch.load(path, map_location=map_location)
    return _extract_state_dict(payload), meta


def load_training_checkpoint(
    path: str | Path,
    expected: dict | None = None,
    map_location="cpu",
    *,
    require_meta_schema: bool = False,
):
    """加载完整训练状态；新 checkpoint 校验绑定，历史 sidecar 保持可迁移 resume。"""

    meta = load_meta(path, require_schema=require_meta_schema)
    assert_checkpoint_config(meta, expected)
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or "model_state" not in payload or "optimizer_state" not in payload:
        raise ValueError(f"{path} 不是完整训练 checkpoint，无法 resume")
    if "epoch" not in payload:
        raise ValueError(f"{path} 缺少 epoch，无法 resume")
    return payload, meta
