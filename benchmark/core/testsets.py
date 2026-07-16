"""统一 benchmark testset 清单的读取、指纹计算与数据泄漏验证。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "benchmark" / "testsets.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class TestsetSpec:
    """描述一个可复现的数据 split 及其评估输入契约。"""

    id: str
    family: str
    dataset_version: str
    split: str
    manifest: str
    feature_mapping: str | None
    input_dim: int | None
    labels: tuple[str, ...]
    purpose: str
    root: Path = field(repr=False)
    data_root: str | None = None
    expected_items: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


def resolve_path(value: str | Path, root: str | Path = REPO_ROOT) -> Path:
    """把清单中的路径解析为绝对路径；绝对路径保持不变。"""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(root).expanduser().resolve() / path).resolve()


def load_testsets(
    path: str | Path = DEFAULT_CATALOG,
    *,
    root: str | Path | None = None,
) -> dict[str, TestsetSpec]:
    """读取 testset YAML，并按稳定 id 返回规格索引。"""

    catalog_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"testset catalog 必须是 YAML mapping: {catalog_path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"不支持的 testset schema_version: {payload.get('schema_version')!r}")
    entries = payload.get("testsets")
    if not isinstance(entries, dict):
        raise ValueError("testset catalog 缺少 `testsets` mapping")

    catalog_root = Path(root).resolve() if root is not None else resolve_path(payload.get("root", "."), catalog_path.parent)
    specs: dict[str, TestsetSpec] = {}
    for testset_id, item in entries.items():
        if not isinstance(item, dict):
            raise ValueError(f"testset {testset_id!r} 必须是 mapping")
        input_dim = item.get("input_dim")
        specs[str(testset_id)] = TestsetSpec(
            id=str(testset_id),
            family=str(item.get("family", "")),
            dataset_version=str(item.get("dataset_version", "")),
            split=str(item.get("split", "")),
            manifest=str(item.get("manifest", "")),
            feature_mapping=None if item.get("feature_mapping") is None else str(item["feature_mapping"]),
            input_dim=input_dim if isinstance(input_dim, int) and not isinstance(input_dim, bool) else None,
            labels=tuple(str(label) for label in item.get("labels", []) if str(label)),
            purpose=str(item.get("purpose", "")),
            root=catalog_root,
            data_root=None if item.get("data_root") is None else str(item["data_root"]),
            expected_items=tuple(str(name) for name in item.get("expected_items", []) if str(name)),
            raw=dict(item),
        )
    return specs


def get_testset(
    testset_id: str,
    catalog: Mapping[str, TestsetSpec] | None = None,
    *,
    path: str | Path = DEFAULT_CATALOG,
    root: str | Path | None = None,
) -> TestsetSpec:
    """按 id 读取一个 testset；未知 id 直接报出可选项。"""

    specs = dict(catalog) if catalog is not None else load_testsets(path, root=root)
    try:
        return specs[testset_id]
    except KeyError as exc:
        choices = ", ".join(sorted(specs)) or "<empty>"
        raise KeyError(f"未知 testset: {testset_id}; 可选: {choices}") from exc


def _load_yaml(path: Path) -> dict[str, Any]:
    """读取需要为 mapping 的 YAML 文件。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML 必须是 mapping: {path}")
    return payload


def _yolo_data_root(data_yaml: Path, payload: Mapping[str, Any]) -> Path:
    """按 Ultralytics data.yaml 语义解析数据集根目录。"""

    configured = payload.get("path")
    if configured is None:
        return data_yaml.parent.resolve()
    path = Path(str(configured)).expanduser()
    return path.resolve() if path.is_absolute() else (data_yaml.parent / path).resolve()


def _read_yolo_image_paths(data_yaml: Path, split: str) -> list[Path]:
    """读取 YOLO 某个 split 的图片路径，兼容目录、图片和 txt 清单。"""

    payload = _load_yaml(data_yaml)
    data_root = _yolo_data_root(data_yaml, payload)
    configured = payload.get(split)
    if configured is None:
        raise ValueError(f"{data_yaml} 未声明 split={split}")
    entries = configured if isinstance(configured, list) else [configured]
    images: list[Path] = []
    for entry in entries:
        path = Path(str(entry)).expanduser()
        path = path.resolve() if path.is_absolute() else (data_root / path).resolve()
        if path.is_dir():
            images.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() == ".txt":
            for raw in path.read_text(encoding="utf-8").splitlines():
                value = raw.strip()
                if not value:
                    continue
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    candidate = data_root / candidate
                images.append(candidate.resolve())
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise FileNotFoundError(f"YOLO split={split} 路径不存在或格式不支持: {path}")
    return images


def read_split_items(spec: TestsetSpec) -> list[str]:
    """读取 testset 的稳定样本标识，不加载大体量特征或图片内容。"""

    manifest = resolve_path(spec.manifest, spec.root)
    if spec.family == "temporal":
        return [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if spec.family == "yolo":
        data = _load_yaml(manifest)
        data_root = _yolo_data_root(manifest, data)
        items = []
        for image in _read_yolo_image_paths(manifest, spec.split):
            try:
                items.append(image.relative_to(data_root).as_posix())
            except ValueError:
                items.append(str(image))
        return items
    if spec.family == "e2e":
        case = _load_yaml(manifest)
        case_id = case.get("case_id")
        return [str(case_id)] if case_id else []
    raise ValueError(f"{spec.id}: 不支持 family={spec.family!r}")


def manifest_fingerprint(spec: TestsetSpec) -> str:
    """计算包含规格、manifest 内容和样本清单的 SHA-256 指纹。"""

    manifest = resolve_path(spec.manifest, spec.root)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    payload = {
        "id": spec.id,
        "family": spec.family,
        "dataset_version": spec.dataset_version,
        "split": spec.split,
        "feature_mapping": spec.feature_mapping,
        "input_dim": spec.input_dim,
        "labels": list(spec.labels),
        "purpose": spec.purpose,
        "manifest_sha256": manifest_sha256,
        "items": read_split_items(spec),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_video_id(image: Path) -> str:
    """从抽帧文件名恢复源视频 id，用于检测跨 split 泄漏。"""

    stem = image.stem
    if stem.startswith("ms_"):
        stem = stem[3:]
    video_match = re.match(r"^(.*)\.(?:mp4|avi|mov|mkv|m4v)-\d+$", stem, flags=re.IGNORECASE)
    if video_match:
        return video_match.group(1)
    return re.sub(r"(?:-|_)(?:frame[-_]?)?\d{4,}$", "", stem, flags=re.IGNORECASE)


def _validate_required_fields(spec: TestsetSpec) -> list[str]:
    """验证所有模型族共享的 testset 元数据。"""

    errors = []
    if spec.family not in {"temporal", "yolo", "e2e"}:
        errors.append(f"family 非法: {spec.family!r}")
    for name, value in (
        ("dataset_version", spec.dataset_version),
        ("split", spec.split),
        ("manifest", spec.manifest),
        ("purpose", spec.purpose),
    ):
        if not value:
            errors.append(f"缺少必填字段: {name}")
    if not spec.feature_mapping:
        errors.append("缺少必填字段: feature_mapping")
    if spec.family in {"temporal", "yolo"} and (spec.input_dim is None or spec.input_dim <= 0):
        errors.append("input_dim 必须是正整数")
    if not spec.labels:
        errors.append("labels 不能为空")
    if len(spec.labels) != len(set(spec.labels)):
        errors.append("labels 含重复项")
    return errors


def _validate_temporal(spec: TestsetSpec) -> list[str]:
    """验证时序 split 清单、标签映射以及每个样本的特征和真值文件。"""

    errors = []
    if not spec.data_root:
        return ["temporal testset 缺少 data_root"]
    data_root = resolve_path(spec.data_root, spec.root)
    try:
        items = read_split_items(spec)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    if not items:
        errors.append("split 清单为空")
    if len(items) != len(set(items)):
        errors.append("split 清单含重复样本")
    if spec.expected_items and tuple(items) != spec.expected_items:
        errors.append(f"split 样本与 expected_items 不一致: actual={items!r}")

    if spec.raw.get("format") == "actionmixed_bbox":
        mapping_path = data_root / "labels" / "data.yaml"
        if not mapping_path.is_file():
            errors.append(f"缺少动作标签映射: {mapping_path}")
        else:
            payload = _load_yaml(mapping_path)
            names = payload.get("names") or {}
            mapped = tuple(str(names[key]) for key in sorted(names, key=lambda value: int(value)))
            if mapped != spec.labels:
                errors.append(f"ActionMixed labels 与 manifest 不一致: {mapped!r}")
        for name in items:
            label_path = data_root / "labels" / spec.split / f"{name}.txt"
            if not label_path.is_file():
                errors.append(f"缺少动作标签文件: {label_path}")
        return errors

    mapping_path = data_root / "mapping.txt"
    if not mapping_path.is_file():
        errors.append(f"缺少标签映射: {mapping_path}")
    else:
        mapped: list[tuple[int, str]] = []
        try:
            for line in mapping_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    index, label = line.split(maxsplit=1)
                    mapped.append((int(index), label))
            mapping_labels = tuple(label for _, label in sorted(mapped))
            if mapping_labels != spec.labels:
                errors.append(f"mapping labels 与 manifest 不一致: {mapping_labels!r}")
        except (TypeError, ValueError) as exc:
            errors.append(f"无法解析标签映射 {mapping_path}: {exc}")

    for name in items:
        feature = data_root / "features" / f"{name}.npy"
        truth = data_root / "groundTruth" / f"{name}.txt"
        if not feature.is_file():
            errors.append(f"缺少特征文件: {feature}")
        if not truth.is_file():
            errors.append(f"缺少标签文件: {truth}")
    return errors


def _validate_yolo(spec: TestsetSpec) -> list[str]:
    """验证 YOLO data.yaml，并按源视频 id 检测 train/val/test 泄漏。"""

    errors = []
    manifest = resolve_path(spec.manifest, spec.root)
    if not manifest.is_file():
        return [f"缺少 YOLO data.yaml: {manifest}"]
    try:
        payload = _load_yaml(manifest)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [str(exc)]

    names = payload.get("names") or {}
    if isinstance(names, dict):
        parsed_labels = tuple(str(names[key]) for key in sorted(names, key=lambda value: int(value)))
    elif isinstance(names, list):
        parsed_labels = tuple(str(value) for value in names)
    else:
        parsed_labels = ()
    if parsed_labels != spec.labels:
        errors.append(f"data.yaml labels 与 manifest 不一致: {parsed_labels!r}")

    videos: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        try:
            images = _read_yolo_image_paths(manifest, split)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            videos[split] = set()
            continue
        if not images:
            errors.append(f"YOLO split={split} 没有图片")
        videos[split] = {_source_video_id(image) for image in images}

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(videos[left] & videos[right])
        if overlap:
            errors.append(f"YOLO 源视频跨 split 泄漏 {left}/{right}: {overlap}")
    return errors


def _validate_e2e(spec: TestsetSpec) -> list[str]:
    """验证端到端 case 的业务期望和阶段时间范围。"""

    manifest = resolve_path(spec.manifest, spec.root)
    if not manifest.is_file():
        return [f"缺少 e2e case: {manifest}"]
    try:
        case = _load_yaml(manifest)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [str(exc)]

    errors = []
    if not case.get("case_id"):
        errors.append("e2e case 缺少 case_id")
    if not case.get("video"):
        errors.append("e2e case 缺少 video")
    try:
        duration = float(case.get("duration_sec", 0))
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        errors.append("e2e duration_sec 必须大于 0")

    expected = case.get("expected")
    if not isinstance(expected, dict) or not expected.get("result"):
        return errors + ["e2e case 缺少 expected.result"]
    required_actions = expected.get("required_actions") or []
    if not isinstance(required_actions, list) or not required_actions:
        errors.append("e2e required_actions 不能为空")
    unknown_required = sorted(set(map(str, required_actions)) - set(spec.labels))
    if unknown_required:
        errors.append(f"required_actions 不在 manifest labels 中: {unknown_required}")

    phases = expected.get("phases") or []
    if not isinstance(phases, list) or not phases:
        errors.append("e2e phases 不能为空")
        return errors
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            errors.append(f"phase[{index}] 必须是 mapping")
            continue
        name = str(phase.get("name", ""))
        if name not in spec.labels:
            errors.append(f"phase[{index}] label 未登记: {name!r}")
        try:
            start = float(phase["start_sec"])
            end = float(phase["end_sec"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"phase[{index}] 缺少合法 start_sec/end_sec")
            continue
        if start < 0 or end <= start or (duration > 0 and end > duration):
            errors.append(f"phase[{index}] 时间范围非法: start={start}, end={end}, duration={duration}")
    return errors


def validate_spec(spec: TestsetSpec) -> list[str]:
    """验证单个 testset，返回可直接展示的错误列表。"""

    errors = _validate_required_fields(spec)
    manifest = resolve_path(spec.manifest, spec.root) if spec.manifest else None
    if manifest is not None and not manifest.exists():
        errors.append(f"manifest 不存在: {manifest}")
        return list(dict.fromkeys(errors))
    if spec.family == "temporal":
        errors.extend(_validate_temporal(spec))
    elif spec.family == "yolo":
        errors.extend(_validate_yolo(spec))
    elif spec.family == "e2e":
        errors.extend(_validate_e2e(spec))
    return list(dict.fromkeys(errors))


def validate_catalog(catalog: Mapping[str, TestsetSpec]) -> dict[str, list[str]]:
    """验证整个清单，并额外检查同版本 temporal train/test 互斥。"""

    results = {testset_id: validate_spec(spec) for testset_id, spec in catalog.items()}
    temporal_groups: dict[str, list[TestsetSpec]] = {}
    for spec in catalog.values():
        if spec.family == "temporal":
            temporal_groups.setdefault(spec.dataset_version, []).append(spec)

    for dataset_version, specs in temporal_groups.items():
        trains = [spec for spec in specs if spec.split == "train"]
        tests = [spec for spec in specs if spec.split == "test"]
        if not trains or not tests:
            message = f"temporal dataset_version={dataset_version} 必须同时登记 train 和 test"
            for spec in specs:
                results[spec.id].append(message)
            continue
        split_order = {"train": 0, "val": 1, "test": 2}
        ordered = sorted(specs, key=lambda item: (split_order.get(item.split, 99), item.id))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if left.split == right.split:
                    continue
                try:
                    overlap = sorted(set(read_split_items(left)) & set(read_split_items(right)))
                except (OSError, ValueError):
                    continue
                if overlap:
                    message = (
                        f"temporal {left.split}/{right.split} 泄漏 "
                        f"dataset_version={dataset_version}: {overlap}"
                    )
                    results[left.id].append(message)
                    results[right.id].append(message)

    return {testset_id: list(dict.fromkeys(errors)) for testset_id, errors in results.items()}
