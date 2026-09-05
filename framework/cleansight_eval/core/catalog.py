"""统一 catalog（数据契约 + testset 口径）的读取、指纹计算与数据泄漏验证。

catalog 是训练与评测共享的数据契约层：``datasets:`` 段登记数据集定义（数据根、labels、
feature_mapping、input_dim），``testsets:`` 段登记评测口径（split、purpose、expected_items）。
framework 训练时用它解析 ``data.dataset_ref``，benchmark 评测时用它钉定固定 testset。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = REPO_ROOT / "framework" / "testsets.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_OVERLAP_POLICIES = {"error", "frame", "allow"}


@dataclass(frozen=True)
class TestsetSpec:
    """描述一个可复现的数据 split 及其评估输入契约。"""

    id: str
    dataset: str | None
    family: str
    dataset_version: str
    dataset_revision: str | None
    split: str
    manifest: str
    feature_mapping: str | None
    input_dim: int | None
    labels: tuple[str, ...]
    purpose: str
    split_overlap_policy: str
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
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError(f"不支持的 testset schema_version: {schema_version!r}")
    datasets = payload.get("datasets", {})
    if schema_version == 2 and not isinstance(datasets, dict):
        raise ValueError("testset catalog v2 缺少 `datasets` mapping")
    entries = payload.get("testsets")
    if not isinstance(entries, dict):
        raise ValueError("testset catalog 缺少 `testsets` mapping")

    catalog_root = (
        Path(root).resolve()
        if root is not None
        else resolve_path(payload.get("root", "."), catalog_path.parent)
    )
    specs: dict[str, TestsetSpec] = {}
    for testset_id, item in entries.items():
        if not isinstance(item, dict):
            raise ValueError(f"testset {testset_id!r} 必须是 mapping")
        dataset_id = None
        resolved = dict(item)
        if schema_version == 2:
            dataset_id = item.get("dataset")
            if not isinstance(dataset_id, str) or not dataset_id:
                raise ValueError(f"testset {testset_id!r} 缺少 `dataset` 引用")
            dataset = datasets.get(dataset_id)
            if not isinstance(dataset, dict):
                raise ValueError(f"testset {testset_id!r} 引用了未知 dataset: {dataset_id!r}")
            duplicate_fields = sorted(set(dataset) & (set(item) - {"dataset"}))
            if duplicate_fields:
                raise ValueError(
                    f"testset {testset_id!r} 重复声明 dataset 公共字段: {duplicate_fields}"
                )
            resolved = {**dataset, **{key: value for key, value in item.items() if key != "dataset"}}
        input_dim = resolved.get("input_dim")
        specs[str(testset_id)] = TestsetSpec(
            id=str(testset_id),
            dataset=dataset_id,
            family=str(resolved.get("family", "")),
            dataset_version=str(resolved.get("dataset_version", "")),
            dataset_revision=(
                None
                if resolved.get("revision") is None
                else str(resolved["revision"])
            ),
            split=str(resolved.get("split", "")),
            manifest=str(resolved.get("manifest", "")),
            feature_mapping=(
                None
                if resolved.get("feature_mapping") is None
                else str(resolved["feature_mapping"])
            ),
            input_dim=input_dim if isinstance(input_dim, int) and not isinstance(input_dim, bool) else None,
            labels=tuple(str(label) for label in resolved.get("labels", []) if str(label)),
            purpose=str(resolved.get("purpose", "")),
            split_overlap_policy=str(resolved.get("split_overlap_policy", "error")),
            root=catalog_root,
            data_root=None if resolved.get("data_root") is None else str(resolved["data_root"]),
            expected_items=tuple(str(name) for name in resolved.get("expected_items", []) if str(name)),
            raw=resolved,
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


def get_dataset_split(
    dataset_ref: str,
    split: str,
    catalog: Mapping[str, TestsetSpec] | None = None,
) -> TestsetSpec:
    """按数据集引用和 split 返回唯一登记规格，避免训练端自行猜测 manifest。"""

    specs = dict(catalog) if catalog is not None else load_testsets()
    matches = [
        spec
        for spec in specs.values()
        if spec.dataset == dataset_ref and spec.split == split
    ]
    if len(matches) != 1:
        choices = sorted(spec.id for spec in matches)
        raise KeyError(
            f"dataset_ref={dataset_ref!r} split={split!r} 必须唯一登记，"
            f"当前匹配 {choices or '<empty>'}"
        )
    return matches[0]


def get_dataset_specs(
    dataset_ref: str,
    catalog: Mapping[str, TestsetSpec] | None = None,
) -> list[TestsetSpec]:
    """返回一个数据集引用下的全部 split 规格，并校验公共契约没有漂移。"""

    specs = dict(catalog) if catalog is not None else load_testsets()
    matches = sorted(
        (spec for spec in specs.values() if spec.dataset == dataset_ref),
        key=lambda spec: (spec.split, spec.id),
    )
    if not matches:
        raise KeyError(f"未知 dataset_ref: {dataset_ref!r}")
    baseline = matches[0]
    for spec in matches[1:]:
        for field_name in (
            "family",
            "dataset_version",
            "dataset_revision",
            "data_root",
            "feature_mapping",
            "input_dim",
            "labels",
            "split_overlap_policy",
        ):
            if getattr(spec, field_name) != getattr(baseline, field_name):
                raise ValueError(
                    f"dataset_ref={dataset_ref!r} 的公共字段 {field_name} 在 split 间不一致"
                )
    return matches


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


def _actionmixed_content_fingerprint(spec: TestsetSpec) -> str:
    """计算该 split 训练实际消费的动作标签、检测映射和逐帧 bbox 内容摘要。"""

    if not spec.data_root:
        raise ValueError(f"{spec.id}: ActionMixed 缺少 data_root")
    data_root = resolve_path(spec.data_root, spec.root)
    digest = hashlib.sha256()

    def update_file(path: Path) -> None:
        try:
            relative = path.relative_to(data_root).as_posix()
        except ValueError:
            relative = str(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if not path.is_file():
            digest.update(b"<missing>\0")
            return
        digest.update(hashlib.sha256(path.read_bytes()).digest())

    update_file(data_root / "labels" / "data.yaml")
    update_file(data_root / "frames" / "data.yaml")
    for name in read_split_items(spec):
        label_path = data_root / "labels" / spec.split / f"{name}.txt"
        update_file(label_path)
        if not label_path.is_file():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                frame_id = int(parts[0])
            except ValueError:
                continue
            update_file(
                data_root / "frames" / spec.split / f"{name}-{frame_id:06d}.txt"
            )
    return digest.hexdigest()


def manifest_fingerprint(spec: TestsetSpec) -> str:
    """计算包含规格、manifest 内容和样本清单的 SHA-256 指纹。"""

    manifest = resolve_path(spec.manifest, spec.root)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    payload = {
        "id": spec.id,
        "dataset": spec.dataset,
        "family": spec.family,
        "dataset_version": spec.dataset_version,
        "split": spec.split,
        "feature_mapping": spec.feature_mapping,
        "input_dim": spec.input_dim,
        "labels": list(spec.labels),
        "purpose": spec.purpose,
        "split_overlap_policy": spec.split_overlap_policy,
        "manifest_sha256": manifest_sha256,
        "items": read_split_items(spec),
    }
    if spec.dataset_revision is not None:
        payload["dataset_revision"] = spec.dataset_revision
    if spec.family == "temporal" and spec.raw.get("format") == "actionmixed_bbox":
        payload["content_sha256"] = _actionmixed_content_fingerprint(spec)
        if spec.split_overlap_policy == "frame":
            payload["frame_keys"] = sorted(_read_actionmixed_frame_keys(spec))
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
    return re.sub(r"(?:-|_)(?:frame[-_]?)?\d{4,}(?:_dense)?$", "", stem, flags=re.IGNORECASE)


def _source_frame_key(image: Path) -> tuple[str, int] | None:
    """从抽帧文件名恢复 ``(源视频ID, 帧ID)``，无法识别帧号时返回 ``None``。

    兼容稀有类相邻帧密采的 ``_dense`` 后缀（如 ``t85_000843_dense.jpg``），
    密采帧仍归属同一 ``(源视频ID, 帧ID)`` 键，用于跨 split 帧级泄漏检查。
    """

    stem = image.stem
    if stem.startswith("ms_"):
        stem = stem[3:]
    video_match = re.match(
        r"^(.*)\.(?:mp4|avi|mov|mkv|m4v)-(\d+)$", stem, flags=re.IGNORECASE
    )
    if video_match:
        return video_match.group(1), int(video_match.group(2))
    frame_match = re.match(
        r"^(.*?)(?:-|_)(?:frame[-_]?)?(\d{4,})(?:_dense)?$", stem, flags=re.IGNORECASE
    )
    if frame_match:
        return frame_match.group(1), int(frame_match.group(2))
    return None


def _read_actionmixed_frame_keys(spec: TestsetSpec) -> set[tuple[str, int]]:
    """读取 ActionMixed split 的 ``(视频名, 帧ID)``，供帧级跨集合门禁比较。"""

    if spec.raw.get("format") != "actionmixed_bbox" or not spec.data_root:
        raise ValueError(f"{spec.id}: frame 策略仅支持带 data_root 的 actionmixed_bbox")
    data_root = resolve_path(spec.data_root, spec.root)
    keys: set[tuple[str, int]] = set()
    for name in read_split_items(spec):
        label_path = data_root / "labels" / spec.split / f"{name}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                frame_id = int(parts[0])
            except ValueError:
                continue
            keys.add((name, frame_id))
    return keys


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
    if spec.split_overlap_policy not in SPLIT_OVERLAP_POLICIES:
        errors.append(
            "split_overlap_policy 必须是 error、frame 或 allow，"
            f"当前为 {spec.split_overlap_policy!r}"
        )
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
        split_dir = data_root / "labels" / spec.split
        actual_items = (
            sorted(path.name[:-4] for path in split_dir.glob("*.txt"))
            if split_dir.is_dir()
            else []
        )
        registered = set(items)
        actual = set(actual_items)
        missing_from_manifest = sorted(actual - registered)
        missing_from_split = sorted(registered - actual)
        if missing_from_manifest:
            errors.append(f"split 目录存在未登记样本: {missing_from_manifest}")
        if missing_from_split:
            errors.append(f"manifest 样本不在 split 目录: {missing_from_split}")

        mapping_path = data_root / "labels" / "data.yaml"
        if not mapping_path.is_file():
            errors.append(f"缺少动作标签映射: {mapping_path}")
        else:
            payload = _load_yaml(mapping_path)
            names = payload.get("names") or {}
            mapped = tuple(str(names[key]) for key in sorted(names, key=lambda value: int(value)))
            if mapped != spec.labels:
                errors.append(f"ActionMixed labels 与 manifest 不一致: {mapped!r}")

        detection_mapping_path = data_root / "frames" / "data.yaml"
        if not detection_mapping_path.is_file():
            errors.append(f"缺少检测目标映射: {detection_mapping_path}")
        else:
            payload = _load_yaml(detection_mapping_path)
            detection_names = payload.get("names") or {}
            if isinstance(detection_names, dict):
                detection_count = len(detection_names)
            elif isinstance(detection_names, list):
                detection_count = len(detection_names)
            else:
                detection_count = 0
            if detection_count * 5 != spec.input_dim:
                errors.append(
                    f"ActionMixed 检测类别数×5={detection_count * 5} "
                    f"与 input_dim={spec.input_dim} 不一致"
                )

        missing_bbox: list[str] = []
        for name in items:
            label_path = data_root / "labels" / spec.split / f"{name}.txt"
            if not label_path.is_file():
                errors.append(f"缺少动作标签文件: {label_path}")
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 2:
                    continue
                try:
                    frame_id = int(parts[0])
                except ValueError:
                    continue
                frame_path = data_root / "frames" / spec.split / f"{name}-{frame_id:06d}.txt"
                if not frame_path.is_file():
                    missing_bbox.append(frame_path.relative_to(data_root).as_posix())
        if missing_bbox:
            preview = missing_bbox[:5]
            errors.append(
                f"缺少 {len(missing_bbox)} 个逐帧 bbox 文件，示例: {preview}"
            )
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
    frames: dict[str, set[tuple[str, int]]] = {}
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
        frame_keys = [_source_frame_key(image) for image in images]
        frames[split] = {key for key in frame_keys if key is not None}
        if spec.split_overlap_policy == "frame" and len(frames[split]) != len(images):
            errors.append(f"YOLO split={split} 存在无法恢复源视频/帧ID的图片")

    if spec.split_overlap_policy == "error":
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sorted(videos[left] & videos[right])
            if overlap:
                errors.append(f"YOLO 源视频跨 split 泄漏 {left}/{right}: {overlap}")
    elif spec.split_overlap_policy == "frame":
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = sorted(frames[left] & frames[right])
            if overlap:
                errors.append(f"YOLO 具体帧跨 split 泄漏 {left}/{right}: {overlap}")
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
        policies = {spec.split_overlap_policy for spec in specs}
        if len(policies) != 1:
            message = (
                f"temporal dataset_version={dataset_version} 的 split_overlap_policy 不一致: "
                f"{sorted(policies)}"
            )
            for spec in specs:
                results[spec.id].append(message)
            continue
        overlap_policy = next(iter(policies))
        trains = [spec for spec in specs if spec.split == "train"]
        tests = [spec for spec in specs if spec.split == "test"]
        # 2026-09-05 起 auto 通道取消 test 集：train 为必须项，test 变为可选
        # （登记 test 仍执行后续帧级/样本级泄漏检查；未登记则该版本跳过泄漏检查）。
        vals = [spec for spec in specs if spec.split == "val"]
        if not trains or not (tests or vals):
            message = f"temporal dataset_version={dataset_version} 必须登记 train，且至少有 test 或 val 之一"
            for spec in specs:
                results[spec.id].append(message)
            continue
        if not tests:
            continue
        split_order = {"train": 0, "val": 1, "test": 2}
        ordered = sorted(specs, key=lambda item: (split_order.get(item.split, 99), item.id))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if left.split == right.split:
                    continue
                try:
                    if overlap_policy == "frame":
                        overlap = sorted(
                            _read_actionmixed_frame_keys(left)
                            & _read_actionmixed_frame_keys(right)
                        )
                    else:
                        overlap = sorted(set(read_split_items(left)) & set(read_split_items(right)))
                except (OSError, ValueError) as exc:
                    if overlap_policy == "frame":
                        message = (
                            f"temporal dataset_version={dataset_version} 无法执行帧级泄漏检查: {exc}"
                        )
                        results[left.id].append(message)
                        results[right.id].append(message)
                    continue
                if overlap and overlap_policy != "allow":
                    overlap_name = "帧泄漏" if overlap_policy == "frame" else "泄漏"
                    message = (
                        f"temporal {left.split}/{right.split} {overlap_name} "
                        f"dataset_version={dataset_version}: {overlap}"
                    )
                    results[left.id].append(message)
                    results[right.id].append(message)

    return {testset_id: list(dict.fromkeys(errors)) for testset_id, errors in results.items()}
