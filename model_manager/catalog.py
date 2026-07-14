"""读取并校验模型集的唯一模型清单。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path(__file__).with_name("models.yaml")


@dataclass(frozen=True)
class ModelSpec:
    """一个已登记模型及其输入、权重、工厂和评估契约。"""

    id: str
    family: str
    adapter: str
    workdir: Path
    target: str
    raw: dict[str, Any]

    def _output_path(self, key: str) -> Path | None:
        value = self.raw.get("output", {}).get(key)
        return self.workdir / value if value else None

    @property
    def checkpoint(self) -> Path | None:
        """返回登记的 checkpoint 绝对路径。"""

        return self._output_path("checkpoint")

    @property
    def report(self) -> Path | None:
        """返回登记的评估报告绝对路径。"""

        return self._output_path("report")

    @property
    def card(self) -> Path | None:
        """返回登记的 CARD.md 绝对路径。"""

        return self._output_path("card")

    @property
    def pin(self) -> Path | None:
        """返回登记的 pin.yaml 绝对路径。"""

        return self._output_path("pin")

    @property
    def checkpoint_sha256(self) -> str | None:
        """返回清单钉定的 checkpoint SHA256。"""

        value = self.raw.get("output", {}).get("checkpoint_sha256")
        return str(value) if value else None

    @property
    def evaluation(self) -> dict[str, Any]:
        """返回该模型的统一评估配置。"""

        return dict(self.raw.get("evaluation", {}))

    @property
    def factory(self) -> dict[str, Any]:
        """返回动态构造模型所需的模块文件和类名。"""

        return dict(self.raw.get("factory", {}))


def load_catalog(path: Path = CATALOG) -> dict[str, Any]:
    """读取模型 YAML；空文件规范为空字典。"""

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并 profile 与模型条目；模型条目优先。"""

    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _template_context(item: dict[str, Any]) -> dict[str, str]:
    output = item.get("output", {}) if isinstance(item.get("output"), dict) else {}
    return {
        "id": str(item.get("id", "")),
        "target": str(item.get("target", "")),
        "workdir": str(item.get("workdir", "")),
        "checkpoint": str(output.get("checkpoint", "")),
    }


def _expand_templates(value: Any, context: dict[str, str]) -> Any:
    """展开 YAML 字符串中的简单占位符，保持列表/字典结构不变。"""

    if isinstance(value, str):
        return value.format_map(context)
    if isinstance(value, list):
        return [_expand_templates(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _expand_templates(item, context) for key, item in value.items()}
    return value


def resolve_model_item(item: dict[str, Any], profiles: dict[str, Any]) -> dict[str, Any]:
    """把单个模型条目与 profile 合并成完整配置。"""

    profile_name = item.get("profile")
    if profile_name:
        if profile_name not in profiles:
            raise ValueError(f"未知模型 profile: {profile_name}")
        profile = profiles[profile_name]
        if not isinstance(profile, dict):
            raise ValueError(f"profile 必须是 mapping: {profile_name}")
        merged = _deep_merge(profile, item)
    else:
        merged = dict(item)
    return _expand_templates(merged, _template_context(merged))


def load_models(path: Path = CATALOG) -> dict[str, ModelSpec]:
    """读取模型条目并按 id 建索引，重复 id 直接拒绝。"""

    catalog = load_catalog(path)
    profiles = catalog.get("profiles", {})
    if profiles is None:
        profiles = {}
    if not isinstance(profiles, dict):
        raise ValueError("profiles 必须是 mapping")

    models: dict[str, ModelSpec] = {}
    for item in catalog.get("models", []):
        resolved = resolve_model_item(item, profiles)
        model_id = str(resolved["id"])
        if model_id in models:
            raise ValueError(f"模型 id 重复: {model_id}")
        spec = ModelSpec(
            id=model_id,
            family=str(resolved["family"]),
            adapter=str(resolved["adapter"]),
            workdir=ROOT / str(resolved["workdir"]),
            target=str(resolved["target"]),
            raw=resolved,
        )
        models[model_id] = spec
    return models
