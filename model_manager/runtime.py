"""按 models.yaml 的 factory 配置动态构造模型，消除模型名分支。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from model_manager.catalog import ModelSpec


def _load_module(module_path: Path, model_id: str, import_root: Path) -> ModuleType:
    """隔离加载单个模型模块，并临时开放其仓库内依赖。"""

    if not module_path.exists():
        raise FileNotFoundError(f"{model_id} factory module 不存在: {module_path}")
    module_name = "_cleansight_" + "".join(ch if ch.isalnum() else "_" for ch in model_id)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(import_root))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(import_root))
        except ValueError:
            pass
    return module


def load_model_class(spec: ModelSpec):
    """从模型登记的 `factory.module_file/class_name` 返回模型类。"""

    factory = spec.factory
    module_file = factory.get("module_file")
    class_name = factory.get("class_name")
    if not module_file or not class_name:
        raise ValueError(f"{spec.id} 缺少 factory.module_file/class_name")
    module = _load_module(spec.workdir / str(module_file), spec.id, spec.workdir)
    try:
        return getattr(module, str(class_name))
    except AttributeError as exc:
        raise ImportError(f"{spec.id} factory class 不存在: {class_name}") from exc


def build_registered_model(spec: ModelSpec, **overrides: Any):
    """按登记输入维度和类别数构造 `[B,T,F] -> [B,T,C]` 时序模型。"""

    if spec.family != "temporal":
        raise ValueError(f"动态 torch factory 只用于 temporal，收到 {spec.family}")
    input_cfg = spec.raw.get("input", {})
    input_dim = int(overrides.pop("input_dim", input_cfg["input_dim"]))
    num_classes = int(overrides.pop("num_classes", len(input_cfg["labels"])))
    model_class = load_model_class(spec)
    return model_class(input_dim, num_classes, **overrides)
