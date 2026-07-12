"""checkpoint 保存/加载 + 重建元信息（framework 层）。

需求 §7.1/§7.2/§8.1：checkpoint 必须携带能正确重建模型的信息，并且能确认
自身对应的模型配置；评估时不能因错误配置而静默加载。

实现方式：权重存 ``<path>``，重建元信息存 sidecar ``<path>.meta.json``。
元信息包含 type、模型配置、feature schema、input_dim/num_classes/window、pipeline。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .integrity import assert_checkpoint_config

META_SUFFIX = ".meta.json"


def meta_path_for(checkpoint_path: str | Path) -> Path:
    return Path(str(checkpoint_path) + META_SUFFIX)


def save_checkpoint(path: str | Path, state_dict: dict, meta: dict) -> Path:
    """保存权重与重建元信息。

    ``meta`` 至少应包含 type、input_dim、num_classes；建议补充 model 配置、
    feature_schema、window、pipeline，以便评估阶段无需人工重复填写即可重建模型。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
    meta_path_for(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_meta(path: str | Path) -> dict:
    """读取 checkpoint 的重建元信息；缺失则报错（避免盲加载）。"""

    mp = meta_path_for(path)
    if not mp.exists():
        raise FileNotFoundError(
            f"未找到 checkpoint 元信息 {mp}；无法确认该权重对应的模型配置，拒绝加载。"
        )
    return json.loads(mp.read_text(encoding="utf-8"))


def load_checkpoint(path: str | Path, expected: dict | None = None, map_location="cpu"):
    """加载权重并校验配置兼容性。

    返回 ``(state_dict, meta)``。若 ``expected`` 与元信息不兼容，立即抛
    ``CompatibilityError``，实现"不因错误配置静默加载"。
    """

    meta = load_meta(path)
    assert_checkpoint_config(meta, expected)
    state_dict = torch.load(path, map_location=map_location)
    return state_dict, meta
