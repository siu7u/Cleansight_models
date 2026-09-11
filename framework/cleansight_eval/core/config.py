"""实验配置加载、覆盖与有效性检查（framework 层）。

配置驱动同架构变体（需求 §4.3）：族、规模、任务、执行模式、数据、特征、
训练与评估参数、指标都由 YAML 表达。本模块只做与模型语义无关的加载与结构
校验，不理解具体模型。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# 框架层只校验与模型语义无关的通用字段。feature_schema、train、model.input_dim/
# num_classes 等是**流水线专属**要求，下沉到各 Pipeline.validate_config，否则检测这类
# 无特征向量的流水线连配置都过不了。
# pipeline：本实验属于哪条流水线（detection / full_sequence_temporal /
# sliding_window_temporal）；训练与评估同属一条，输入构造与输出语义一致。
CONFIG_SCHEMA_VERSION = 1
REQUIRED_TOP_KEYS = ("schema_version", "pipeline", "model", "data")
ALLOWED_TOP_KEYS = {
    "schema_version", "pipeline", "model", "data", "feature_schema", "augmentation",
    "evaluation", "train", "_config_provenance",
}

# YOLO 检测训练超参词汇：与 ultralytics 的 model.train() 训练参数一一对应。
# 登记后 train 段可以表达这些超参（core 校验放行），由 detection/yolo.py 透传给
# ultralytics；白名单外的键不会被转发，避免拼写错误被静默吞掉。
YOLO_TRAIN_HPARAMS = frozenset({
    "optimizer", "lr0", "lrf", "momentum", "weight_decay", "nbs",
    "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "cos_lr",
    "box", "cls", "dfl", "label_smoothing", "dropout",
    "freeze", "close_mosaic", "fraction", "multi_scale", "amp", "cache",
    "workers", "seed", "deterministic", "single_cls", "rect",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "perspective", "flipud", "fliplr", "bgr", "mosaic", "mixup", "cutmix",
    "copy_paste", "copy_paste_mode", "auto_augment", "erasing",
})

# 已注册模型和流水线共同支持的配置词汇。新增模型参数时需在这里显式登记，拼写错误因而
# 不会被静默忽略。更具体的必填/互斥规则仍由 Pipeline.validate_config 负责。
KNOWN_SECTION_KEYS = {
    "model": {
        "type", "weights", "imgsz", "allow_missing_meta", "input_dim", "num_classes",
        "hidden", "num_layers", "num_stages", "dropout", "tmse_weight", "tmse_clip",
        "d_model", "nhead", "dim_feedforward", "max_len", "lstm_layers", "tcn_layers",
        "refine_stages", "hidden_dims",
        # roi_classification（特征融合）
        "backbone", "roi_size", "freeze_backbone", "hidden_dim",
    },
    "data": {
        "name", "dataset_ref", "data_yaml", "eval_split", "root", "action_mapping", "labels_dir",
        "frames_dir", "split_train", "split_val", "split_eval", "names", "fps",
        # roi_classification（特征融合）
        "classes", "group_dir", "neg_ratio", "val_split", "dataset_dir",
    },
    "feature_schema": {
        "dim", "version", "class_order", "layout", "normalization", "mask_targets",
        "detection_confidence_default",
    },
    "augmentation": {"target_mask"},
    "evaluation": {
        "mode", "testset_id", "save_predictions", "measure_latency", "latency_warmup",
        "latency_runs", "limits", "conf", "iou", "max_det", "agnostic_nms",
        "visualize", "viz_per_page",
    },
    "train": {
        "epochs", "lr", "batch", "batch_size", "patience", "window", "grad_clip",
        "weight_decay", "resume", "best_metric",
    } | YOLO_TRAIN_HPARAMS,
}

PIPELINE_DEFAULTS = {
    "detection": {
        "model.weights": "yolo11n.pt",
        "model.imgsz": 640,
        "model.allow_missing_meta": False,
        "data.eval_split": "val",
        "evaluation.mode": "formal",
        "evaluation.save_predictions": True,
        "evaluation.visualize": True,
        "evaluation.conf": 0.001,
        "evaluation.iou": 0.7,
        "evaluation.max_det": 300,
        "evaluation.agnostic_nms": False,
    },
    "roi_classification": {
        "model.backbone": "resnet50",
        "model.roi_size": 224,
        "model.freeze_backbone": False,
        "model.hidden_dim": 256,
        "model.dropout": 0.3,
        "data.neg_ratio": 1.0,
        "data.val_split": 0.2,
        "evaluation.mode": "exploratory",
        "evaluation.save_predictions": False,
        "evaluation.visualize": False,
    },
    "full_sequence_temporal": {
        "evaluation.mode": "formal",
        "evaluation.save_predictions": True,
        "evaluation.visualize": True,
    },
    "sliding_window_temporal": {
        "evaluation.mode": "formal",
        "evaluation.save_predictions": True,
        "evaluation.visualize": True,
        "evaluation.measure_latency": True,
        "evaluation.latency_warmup": 20,
        "evaluation.latency_runs": 200,
    },
}


def load_config(path: str | Path) -> dict:
    """读取 YAML 实验配置，只做**格式中立**的框架层通用校验。

    流水线专属校验（feature_schema、input_dim、data_yaml…）由各流水线的
    ``validate_config`` 负责，在 CLI 分派器里于本函数之后调用。core 因此**不 import
    任何流水线**，脊柱不反依赖 temporal/detection。
    """

    cfg_path = Path(path).resolve()
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射: {path}")
    validate_config(data)
    raw_fields = sorted(_leaf_paths(data))
    default_fields = materialize_defaults(data)
    validate_config(data)
    resolve_dataset_reference(data, cfg_path.parent, explicit_root="data.root" in raw_fields)
    resolve_relative_paths(data, cfg_path.parent)
    data["_config_provenance"] = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "source_path": str(cfg_path),
        "raw_fields": raw_fields,
        "default_fields": default_fields,
        "override_fields": [],
    }
    return data


def resolve_relative_paths(cfg: dict, base_dir: Path) -> None:
    """把配置中的本地文件路径按配置文件目录解析成绝对路径。"""

    data = cfg.get("data")
    if not isinstance(data, dict):
        return
    for key in ("data_yaml", "root"):
        value = data.get(key)
        if not isinstance(value, str):
            continue
        p = Path(value).expanduser()
        if not p.is_absolute():
            data[key] = str((base_dir / p).resolve())


def resolve_dataset_reference(cfg: dict, base_dir: Path, *, explicit_root: bool) -> None:
    """把 ``data.dataset_ref`` 解析成唯一 catalog 数据根并校验模型/特征/split 契约。"""

    data = cfg.get("data") or {}
    dataset_ref = data.get("dataset_ref")
    if not dataset_ref:
        return

    from .catalog import (
        get_dataset_specs,
        get_dataset_split,
        resolve_path,
        validate_catalog,
    )

    specs = get_dataset_specs(str(dataset_ref))
    validation = validate_catalog({spec.id: spec for spec in specs})
    errors = [f"{testset_id}: {error}" for testset_id, items in validation.items() for error in items]
    if errors:
        raise ValueError(
            f"dataset_ref={dataset_ref!r} 未通过数据门禁:\n  - " + "\n  - ".join(errors)
        )
    baseline = specs[0]
    if not baseline.data_root:
        raise ValueError(f"dataset_ref={dataset_ref!r} 没有登记 data_root")
    catalog_root = resolve_path(baseline.data_root, baseline.root)

    configured_root = data.get("root")
    if explicit_root and isinstance(configured_root, str):
        path = Path(configured_root).expanduser()
        configured = path.resolve() if path.is_absolute() else (base_dir / path).resolve()
        if configured != catalog_root:
            raise ValueError(
                f"data.root={configured} 与 dataset_ref={dataset_ref!r} "
                f"登记根目录 {catalog_root} 不一致"
            )
    data["root"] = str(catalog_root)
    data["name"] = str(dataset_ref)

    feature_schema = cfg.get("feature_schema") or {}
    configured_version = feature_schema.get("version")
    if configured_version is not None and configured_version != baseline.feature_mapping:
        raise ValueError(
            f"feature_schema.version={configured_version!r} 与 dataset_ref "
            f"feature_mapping={baseline.feature_mapping!r} 不一致"
        )
    configured_dim = feature_schema.get("dim")
    if configured_dim is not None and configured_dim != baseline.input_dim:
        raise ValueError(
            f"feature_schema.dim={configured_dim!r} 与 dataset_ref input_dim="
            f"{baseline.input_dim!r} 不一致"
        )

    model = cfg.get("model") or {}
    if baseline.input_dim is not None and model.get("input_dim") != baseline.input_dim:
        raise ValueError(
            f"model.input_dim={model.get('input_dim')!r} 与 dataset_ref "
            f"input_dim={baseline.input_dim!r} 不一致"
        )
    if baseline.labels and model.get("num_classes") != len(baseline.labels):
        raise ValueError(
            f"model.num_classes={model.get('num_classes')!r} 与 dataset_ref "
            f"labels 数量={len(baseline.labels)} 不一致"
        )

    for key in ("split_train", "split_val", "split_eval"):
        split = data.get(key)
        if split:
            get_dataset_split(str(dataset_ref), str(split))

    evaluation_id = (cfg.get("evaluation") or {}).get("testset_id")
    eval_split = data.get("split_eval") or data.get("eval_split")
    if evaluation_id and eval_split:
        expected = get_dataset_split(str(dataset_ref), str(eval_split))
        if evaluation_id != expected.id:
            raise ValueError(
                f"evaluation.testset_id={evaluation_id!r} 与 dataset_ref/split_eval "
                f"推导结果 {expected.id!r} 不一致"
            )


def validate_config(cfg: dict) -> None:
    """框架层通用结构校验（不含任何流水线专属字段）。"""

    missing = [k for k in REQUIRED_TOP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"配置缺少必要字段: {missing}")
    if cfg.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"不支持配置 schema_version={cfg.get('schema_version')!r}，当前为 {CONFIG_SCHEMA_VERSION}"
        )
    unknown_top = sorted(set(cfg) - ALLOWED_TOP_KEYS)
    if unknown_top:
        raise ValueError(f"配置包含未知顶层字段: {unknown_top}")
    if not isinstance(cfg["pipeline"], str) or not cfg["pipeline"]:
        raise ValueError(
            "pipeline 必须是非空字符串，如 sliding_window_temporal / full_sequence_temporal / detection"
        )
    for section, allowed in KNOWN_SECTION_KEYS.items():
        value = cfg.get(section)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"配置 {section} 必须是映射")
        reject_unknown_keys(value, allowed, section)
    evaluation = cfg.get("evaluation") or {}
    mode = evaluation.get("mode")
    if mode is not None and mode not in {"formal", "exploratory"}:
        raise ValueError("evaluation.mode 必须是 formal 或 exploratory")
    if mode == "formal" and (cfg.get("model") or {}).get("allow_missing_meta"):
        raise ValueError("formal 评估不能启用 model.allow_missing_meta")
    if mode == "formal" and evaluation.get("save_predictions") is False:
        raise ValueError("formal 评估必须启用 evaluation.save_predictions 以保留可追溯预测")
    for key in ("conf", "iou"):
        if key in evaluation and not 0.0 <= float(evaluation[key]) <= 1.0:
            raise ValueError(f"evaluation.{key} 必须在 0..1")
    if "max_det" in evaluation and int(evaluation["max_det"]) <= 0:
        raise ValueError("evaluation.max_det 必须大于 0")


def reject_unknown_keys(value: dict, allowed: set[str], path: str) -> None:
    """拒绝未登记字段，错误信息保留完整点路径便于定位拼写。"""

    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"配置 {path} 包含未知字段: {unknown}")


def _leaf_paths(value: dict, prefix: str = "") -> list[str]:
    """列出配置叶子点路径，供 raw/default/override 来源追踪。"""

    paths: list[str] = []
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            paths.extend(_leaf_paths(item, path))
        else:
            paths.append(path)
    return paths


def materialize_defaults(cfg: dict) -> list[str]:
    """把流水线稳定默认值写入最终配置，并返回实际补入的点路径。"""

    defaults = PIPELINE_DEFAULTS.get(cfg.get("pipeline"), {})
    applied: list[str] = []
    for dotted, value in defaults.items():
        keys = dotted.split(".")
        cur = cfg
        for key in keys[:-1]:
            cur = cur.setdefault(key, {})
        if keys[-1] not in cur:
            cur[keys[-1]] = copy.deepcopy(value)
            applied.append(dotted)
    return sorted(applied)


def apply_overrides(cfg: dict, overrides: list[tuple[str, Any]]) -> dict:
    """把 CLI 传入的通用覆盖项按**点路径**写入配置副本，不改动入参。

    覆盖项是 ``(点路径, 值)`` 序列，如 ``("train.epochs", 5)`` / ``("train.batch", 8)``。
    核心 CLI 因此**不预设任何纵的调参名**——每条纵的 trainer 有各自超参词汇（torch 的
    ``batch_size`` vs ultralytics 的 ``batch``），寻址交给调用方，脊柱只做通用点路径写入。
    """

    out = copy.deepcopy(cfg)
    for dotted, value in overrides:
        if not _known_override_path(dotted):
            raise ValueError(f"未知配置覆盖路径: {dotted}")
        _set_dotted(out, dotted, value)
    provenance = out.setdefault("_config_provenance", {})
    provenance["override_fields"] = sorted(
        set(provenance.get("override_fields", [])) | {path for path, _value in overrides}
    )
    validate_config(out)
    source_path = provenance.get("source_path")
    if source_path:
        raw_fields = set(provenance.get("raw_fields", []))
        override_fields = set(provenance.get("override_fields", []))
        resolve_dataset_reference(
            out,
            Path(source_path).parent,
            explicit_root="data.root" in raw_fields or "data.root" in override_fields,
        )
    return out


def _known_override_path(dotted: str) -> bool:
    """判断 override 是否属于显式配置 schema。"""

    parts = dotted.split(".")
    if len(parts) == 1:
        return parts[0] in ALLOWED_TOP_KEYS
    return len(parts) == 2 and parts[0] in KNOWN_SECTION_KEYS and parts[1] in KNOWN_SECTION_KEYS[parts[0]]


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    """按 ``a.b.c`` 点路径写入嵌套字典，沿途缺失的中间层按需建成 dict。"""

    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value
