"""时序数据与 checkpoint 元信息（两条时序流水线共用）。

按 **features 契约**（bbox→40 维）把原始 cleansight-ActionMixed 数据读成逐帧特征序列。
本模块只负责"读原始数据 + 按 features 契约特征化"；加窗/整段等样本构造由各流水线自持
（见 ``full_sequence_pipeline`` / ``sliding_window_pipeline``）。另提供 checkpoint 重建
元信息的构造助手 ``build_temporal_meta``，两条时序流水线复用同一份口径。

历史 ``legacy-20d-v1`` 数据通过同一入口读取 Endo Project 的 ``features/*.npy``、
``groundTruth/*.txt`` 和 ``mapping.txt``。兼容逻辑只处理数据格式，不再保留独立训练或
推理入口。

真实数据格式（已按 train/val/test 目录切分）：

    <root>/labels/data.yaml                      动作类别（nc / names，6 类）
    <root>/labels/<split>/<video>.mp4.txt        每行 "frame_id action_id"（抽样帧，稀疏）
    <root>/frames/<split>/<video>.mp4-<f:06d>.txt 逐帧 YOLO bbox "class cx cy w h"（8 类）

特征口径（features 契约，version=actionmixed-bbox-8cls-v1）：按 8 类顺序，每类取该帧内
**面积最大的一个框** 编码 ``[presence, cx, cy, w, h]``，缺席则全零；拼成 8×5=40 维。
空 bbox 文件（无检测）→ 全零 40 维。可通过 ``feature_schema.mask_targets`` 指定检测目标
名称或类别 ID，将对应类别的特征块清零；遮罩不改变输入维度和类别顺序。

ROI 契约（version=actionmixed-roi-grid-v1，见 ``features/roi_bbox.py``）：同样按 8 类
顺序，每类把画面按 2×3 网格划分成 6 个区域，输出 6×3=18 维 ``[presence, count,
max_area]``，拼成 144 维；该类其他语义（mask_targets、空帧全零、因果逐帧）与 bbox 契约一致。

手部契约（version=actionmixed-bbox-hand-8cls-v1，见 ``features/hand_bbox.py``）：只编码
中心落在"面积最大 hand 框扩张 1.5 倍"区域内的框，坐标相对该区域归一化，维度仍为
8×5=40；无 hand 框时全零。"全局+手部"契约（version=actionmixed-bbox-global-hand-8cls-v1）
为两者拼接，8×5×2=80 维，左半 40 维为全局编码、右半 40 维为手部编码。

图像 embedding 契约（version=actionmixed-bbox-embed-*，见 ``features/image_embed.py``）：
左侧 40 维 bbox 块口径同上，右侧拼接 ``extract_embeddings.py`` 预计算的逐帧整图 CNN
embedding（按标签行位置对齐，``data.embedding_root`` 指向产物根目录），如
``actionmixed-bbox-embed-mbv3s-v1`` = 40 + 576 = 616 维。该契约下 ``mask_targets`` 与
目标随机遮罩只作用于 bbox 块（图像块无检测类结构）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from .features import (
    CLEAN_FEATURE_DIMS,
    EMBED_BBOX_DIM,
    GLOBAL_HAND_BBOX_VERSION,
    HAND_BBOX_VERSION,
    ROI_FEATURE_VERSION,
    ROI_GRID_V2_VERSION,
    ROI_PRESENCE_VERSION,
    build_clean_bbox_features,
    build_hand_frame_features,
    build_roi_frame_features,
    build_roi_grid_v2_frame_features,
    build_roi_presence_frame_features,
    block_dims_for_version,
    is_image_embed_version,
    load_video_embeddings,
    resolve_embed_dim,
    resolve_embedding_root,
    roi_grid_v2_block_dims,
)

N_DET_CLASSES = 8  # frames/data.yaml 的检测类数（每类 5 维）
FEATURE_DIM = N_DET_CLASSES * 5  # = 40
LEGACY_FEATURE_VERSION = "legacy-20d-v1"


def featurize_frame_bbox(
    txt_path: Path,
    n_classes: int = N_DET_CLASSES,
    mask_target_ids: frozenset[int] = frozenset(),
) -> np.ndarray:
    """一帧 bbox → ``[n_classes*5]`` 特征；指定目标的整组特征保持为零。"""
    feat = np.zeros((n_classes, 5), dtype=np.float32)
    best_area = np.zeros(n_classes, dtype=np.float32)
    if txt_path.exists():
        for line in txt_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            c = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
            if not (0 <= c < n_classes):
                continue
            if c in mask_target_ids:
                continue
            area = w * h
            if area >= best_area[c]:  # 取面积最大框；同面积后者覆盖，确定性
                best_area[c] = area
                feat[c] = (1.0, cx, cy, w, h)
    return feat.reshape(-1)  # [n_classes*5]


def load_action_mapping(root: Path, rel: str) -> dict:
    """读 labels/data.yaml（YAML: nc / names），返回 {action_id: name}。"""
    data = yaml.safe_load((root / rel).read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    return {int(k): v for k, v in names.items()}


def load_detection_mapping(data_cfg: dict) -> dict[int, str]:
    """读取 ``frames/data.yaml``，返回 ActionMixed 检测目标的 ``{id: name}`` 映射。"""
    root = Path(data_cfg["root"])
    frames_dir = data_cfg.get("frames_dir", "frames")
    mapping_path = root / frames_dir / "data.yaml"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"检测目标映射不存在: {mapping_path}")
    data = yaml.safe_load(mapping_path.read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(key): str(name) for key, name in names.items()}
    raise ValueError(f"检测目标映射 names 必须是列表或映射: {mapping_path}")


def resolve_mask_target_ids(data_cfg: dict, feature_schema: dict | None) -> frozenset[int]:
    """把 ``mask_targets`` 的目标名/ID解析为类别 ID；未知或越界目标立即报错。"""
    raw_targets = (feature_schema or {}).get("mask_targets")
    if raw_targets is None or raw_targets == "" or raw_targets == []:
        return frozenset()
    if isinstance(raw_targets, (str, int)) and not isinstance(raw_targets, bool):
        # 字符串支持逗号分隔的多个目标（CLI `-S feature_schema.mask_targets=a,b` 用），
        # 单个名字/ID 保持原样。
        targets = [part.strip() for part in raw_targets.split(",")] if isinstance(raw_targets, str) \
            else [raw_targets]
        targets = [part for part in targets if part]
        if not targets:
            raise ValueError("feature_schema.mask_targets 不能是空字符串")
    elif isinstance(raw_targets, list):
        targets = raw_targets
    else:
        raise ValueError("feature_schema.mask_targets 必须是目标名/类别 ID，或它们组成的列表")

    id2name = load_detection_mapping(data_cfg)
    name2id = {name: target_id for target_id, name in id2name.items()}
    resolved: set[int] = set()
    for target in targets:
        if isinstance(target, bool):
            raise ValueError(f"mask_targets 不支持布尔值: {target!r}")
        if isinstance(target, int):
            target_id = target
        elif isinstance(target, str):
            value = target.strip()
            if value in name2id:
                target_id = name2id[value]
            else:
                try:
                    target_id = int(value)
                except ValueError as exc:
                    available = ", ".join(id2name.values())
                    raise ValueError(
                        f"未知遮罩目标 {target!r}；可用目标: {available}"
                    ) from exc
        else:
            raise ValueError(f"mask_targets 元素必须是目标名或类别 ID: {target!r}")
        if target_id not in id2name:
            raise ValueError(
                f"遮罩目标类别 ID 越界: {target_id}；可用 ID: {sorted(id2name)}"
            )
        resolved.add(target_id)
    return frozenset(resolved)


def resolve_target_mask_augmentation(data_cfg: dict, augmentation: dict | None) -> dict | None:
    """校验并解析 train-only 目标随机遮罩配置。

    当前只支持 ``frame_dropout``：对每个指定目标、每个采样帧独立按 ``probability``
    将对应的类别特征块（bbox 契约 5 维 / ROI 契约 18 维）清零。返回值仅供运行时使用，
    不写入配置。
    """

    if augmentation is None:
        return None
    if not isinstance(augmentation, dict):
        raise ValueError("augmentation 必须是映射")
    unknown_augmentation = sorted(set(augmentation) - {"target_mask"})
    if unknown_augmentation:
        raise ValueError(f"augmentation 包含未知字段: {unknown_augmentation}")
    raw = augmentation.get("target_mask")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("augmentation.target_mask 必须是映射")
    allowed = {"enabled", "strategy", "targets", "probability"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"augmentation.target_mask 包含未知字段: {unknown}")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("augmentation.target_mask.enabled 必须是布尔值")
    strategy = raw.get("strategy", "frame_dropout")
    if strategy != "frame_dropout":
        raise ValueError("augmentation.target_mask.strategy 当前只支持 frame_dropout")
    probability = raw.get("probability", 0.0)
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise ValueError("augmentation.target_mask.probability 必须是 0..1 数值")
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("augmentation.target_mask.probability 必须在 0..1")

    targets = raw.get("targets", [])
    if not isinstance(targets, list):
        raise ValueError("augmentation.target_mask.targets 必须是目标名/类别 ID 列表")
    target_ids = resolve_mask_target_ids(data_cfg, {"mask_targets": targets}) if targets else frozenset()
    if enabled and probability > 0.0 and not target_ids:
        raise ValueError("启用随机目标遮罩且 probability > 0 时 targets 不能为空")
    return {
        "enabled": enabled,
        "strategy": strategy,
        "probability": probability,
        "target_ids": target_ids,
    }


def apply_target_mask_augmentation(
    features: list[np.ndarray],
    data_cfg: dict,
    augmentation: dict | None,
    *,
    seed: int,
    feature_schema: dict | None = None,
) -> list[np.ndarray]:
    """对训练集 ``[T, F]`` 特征应用可复现的逐帧目标随机遮罩。

    每个指定目标按其类别特征块清零（bbox 契约 5 维 / ROI 契约 18 维）；块宽由
    特征维与检测类别数推导。**块宽不等的契约**（``actionmixed-roi-grid-v2``：高频类
    27 维、低频类 3 维）改由 ``features.block_dims_for_version`` 提供显式逐类块宽，
    按"总维 ÷ 类数"推导会静默遮错列。图像 embedding 契约（``feature_schema.version``
    属 ``actionmixed-bbox-embed-*``）下遮罩只作用于左侧 bbox 块（块宽 40 ÷ 检测类数），
    图像块无检测类结构、保持原样——该契约的总维度不是"类数 × 块宽"的整数倍语义，
    按总维度推导会静默错切。同一 seed、相同视频顺序和相同配置产生相同遮罩；
    该函数不应由 val/test 数据路径调用。未启用或概率为零时原样返回输入列表。
    """

    spec = resolve_target_mask_augmentation(data_cfg, augmentation)
    if spec is None or not spec["enabled"] or spec["probability"] == 0.0:
        return features

    detection_mapping = load_detection_mapping(data_cfg)
    n_det_classes = len(detection_mapping)
    embed_mask = is_image_embed_version((feature_schema or {}).get("version"))
    explicit_blocks = block_dims_for_version((feature_schema or {}).get("version"), n_det_classes)
    rng = np.random.default_rng(seed)
    augmented: list[np.ndarray] = []
    for sequence in features:
        masked = sequence.copy()
        if masked.ndim != 2:
            raise ValueError(f"目标随机遮罩要求 [T, F] 特征，实际 shape={masked.shape}")
        if embed_mask:
            if EMBED_BBOX_DIM % n_det_classes != 0:
                raise ValueError(
                    f"图像契约的 bbox 块 {EMBED_BBOX_DIM} 维不是检测类数 {n_det_classes} 的整数倍"
                )
            block = EMBED_BBOX_DIM // n_det_classes  # 每类 bbox 块宽（8 类 → 5）
            maskable_dim = EMBED_BBOX_DIM
            if masked.shape[1] < maskable_dim:
                raise ValueError(
                    f"图像契约特征维至少 {maskable_dim}（bbox 块），实际 {masked.shape[1]}"
                )
        elif explicit_blocks is not None:
            block = 0  # 非均匀契约不使用单一块宽
            maskable_dim = sum(explicit_blocks)
            if masked.shape[1] < maskable_dim:
                raise ValueError(
                    f"块宽不等契约的特征维至少 {maskable_dim}（逐类块宽之和），"
                    f"实际 {masked.shape[1]}"
                )
        else:
            block = masked.shape[1] // n_det_classes  # 每类特征块宽（bbox 契约 5 / ROI 契约 18）
            maskable_dim = masked.shape[1]
            if block * n_det_classes != masked.shape[1]:
                raise ValueError(
                    f"目标随机遮罩需要特征维是检测类数 {n_det_classes} 的整数倍，"
                    f"实际 {masked.shape[1]}"
                )
        for target_id in sorted(spec["target_ids"]):
            if explicit_blocks is not None:
                start = sum(explicit_blocks[:target_id])
                width = explicit_blocks[target_id] if target_id < len(explicit_blocks) else 0
                if width == 0:
                    raise ValueError(
                        f"目标 ID={target_id} 超出块宽表 {explicit_blocks}，无法遮罩"
                    )
            else:
                start = target_id * block
                width = block
                if masked.shape[1] < maskable_dim or maskable_dim < start + width:
                    raise ValueError(
                        f"目标 ID={target_id} 的 {width} 维切片超出特征形状 {tuple(masked.shape)}"
                    )
            dropped = rng.random(masked.shape[0]) < spec["probability"]
            masked[dropped, start : start + width] = 0.0
        augmented.append(masked)
    return augmented


def resolve_image_feature_dim(model_cfg: dict, feature_schema: dict | None) -> int:
    """交叉校验「图像 embedding 契约 ↔ 模型投影头」声明，返回图像块宽度（非图像契约为 0）。

    图像契约必须声明 ``model.image_dim``（尾部图像块宽度，用于投影头），且与
    ``feature_schema.dim - 40`` 一致；非图像契约声明 ``model.image_dim`` 直接报错——
    静默忽略会让模型少一个投影头却照常训练出无意义的对照结果。
    """

    declared = model_cfg.get("image_dim")
    if not is_image_embed_version((feature_schema or {}).get("version")):
        if declared:
            raise ValueError(
                f"model.image_dim={declared!r} 只在图像 embedding 契约下有效，"
                f"当前 feature_schema.version={(feature_schema or {}).get('version')!r}"
            )
        return 0
    expected = resolve_embed_dim(feature_schema)
    if declared != expected:
        raise ValueError(
            f"model.image_dim={declared!r} 与契约声明的图像块宽度 {expected} 不一致"
            f"（feature_schema.dim - {EMBED_BBOX_DIM}）"
        )
    return expected


def _registered_split_items(data_cfg: dict, split: str) -> list[str] | None:
    """dataset_ref 存在时读取唯一登记 manifest；临时/合成数据保持目录遍历兼容。"""

    dataset_ref = data_cfg.get("dataset_ref")
    if not dataset_ref:
        return None
    from ..core.catalog import get_dataset_split, read_split_items

    return read_split_items(get_dataset_split(str(dataset_ref), split))


def _split_label_files(data_cfg: dict, split: str) -> list[Path]:
    """split 的标签文件清单（顺序 = 登记 manifest 顺序或目录 sorted 顺序）。

    ``load_split`` / ``split_video_names`` / :func:`count_split_videos` 共用这一处，
    保证"取前 k 个视频"在三处切的是同一批。
    """

    root = Path(data_cfg["root"])
    labels_dir = root / data_cfg.get("labels_dir", "labels") / split
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels split 目录不存在: {labels_dir}")
    registered_items = _registered_split_items(data_cfg, split)
    if registered_items is not None:
        return [labels_dir / f"{name}.txt" for name in registered_items]
    return sorted(labels_dir.glob("*.txt"))


def count_split_videos(data_cfg: dict, split: str) -> int:
    """split 里的视频数（只数列文件，不读内容）。"""

    return len(_split_label_files(data_cfg, split))


def _iter_split_sequences(data_cfg: dict, split: str, window: int | None = None,
                          limit: int | None = None):
    """按登记 manifest 遍历 split，产出 ``(stem, frame_ids, action_ids)``。

    ``data.dataset_ref`` 存在时，manifest 是唯一样本真源；临时/合成配置没有引用时才按目录
    ``sorted`` 遍历。丢弃无有效 "frame_id action_id" 行的空文件；给了 ``window`` 时跳过
    过短序列。``load_split`` 与 ``split_video_names`` 共用此生成器，保证三者严格对齐。
    ``limit`` 取清单**前 k 个**视频（学习曲线用的训练子采样口径，见
    :func:`resolve_train_video_limit`）。
    """
    label_files = _split_label_files(data_cfg, split)
    if limit is not None:
        label_files = label_files[: max(0, int(limit))]
    for label_file in label_files:
        if not label_file.is_file():
            raise FileNotFoundError(f"manifest 登记的动作标签不存在: {label_file}")
        stem = label_file.name[:-4]  # 去掉 ".txt"，保留 "<video>.mp4"
        frame_ids, action_ids = [], []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            frame_ids.append(int(parts[0]))
            action_ids.append(int(parts[1]))
        if not frame_ids:
            continue
        if window is not None and len(frame_ids) < window:
            print(f"  [skip] {label_file.name}: 采样帧 {len(frame_ids)} < window {window}")
            continue
        yield stem, frame_ids, action_ids


def _load_legacy_mapping(root: Path) -> tuple[dict[str, int], dict[int, str]]:
    """读取 Endo Project ``mapping.txt``，返回双向类别映射。"""

    action_to_id: dict[str, int] = {}
    id_to_action: dict[int, str] = {}
    mapping_path = root / "mapping.txt"
    for raw in mapping_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        index_text, action = raw.split(maxsplit=1)
        index = int(index_text)
        if action in action_to_id or index in id_to_action:
            raise ValueError(f"mapping.txt 含重复类别: {raw!r}")
        action_to_id[action] = index
        id_to_action[index] = action
    if not id_to_action:
        raise ValueError(f"标签映射为空: {mapping_path}")
    return action_to_id, id_to_action


def _load_legacy_endo_split(
    data_cfg: dict,
    split: str,
    *,
    window: int | None,
    feature_schema: dict | None,
    max_videos: int | None = None,
    max_frames: int | None = None,
):
    """读取历史 Endo Project split，规范成 framework 的 ``[T,F]`` 序列列表。"""

    root = Path(data_cfg["root"])
    action_to_id, source_id2name = _load_legacy_mapping(root)
    class_order = (feature_schema or {}).get("class_order")
    if class_order is None:
        id2name = source_id2name
        action_id_remap = {index: index for index in source_id2name}
    else:
        if not isinstance(class_order, list) or set(class_order) != set(source_id2name.values()):
            raise ValueError(
                "feature_schema.class_order 与 legacy mapping 不一致: "
                f"configured={class_order}, dataset={list(source_id2name.values())}"
            )
        name_to_target = {name: index for index, name in enumerate(class_order)}
        action_id_remap = {
            source_id: name_to_target[name] for source_id, name in source_id2name.items()
        }
        id2name = {index: name for index, name in enumerate(class_order)}

    names = _registered_split_items(data_cfg, split)
    if names is None:
        bundle = root / "splits" / f"{split}.split1.bundle"
        if not bundle.is_file():
            raise FileNotFoundError(f"legacy split 清单不存在: {bundle}")
        names = [line.strip() for line in bundle.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_videos is not None:
        names = names[:max_videos]

    expected_dim = int((feature_schema or {}).get("dim", data_cfg.get("input_dim", 20)))
    features: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    accepted_names: list[str] = []
    for name in names:
        feature_path = root / "features" / f"{name}.npy"
        truth_path = root / "groundTruth" / f"{name}.txt"
        raw_features = np.load(feature_path)
        if raw_features.ndim != 2:
            raise ValueError(f"{name}: legacy 特征必须是二维，实际 {raw_features.shape}")
        if raw_features.shape[1] == expected_dim:
            sequence = raw_features
        elif raw_features.shape[0] == expected_dim:
            sequence = raw_features.T
        else:
            raise ValueError(
                f"{name}: legacy 特征无法对齐 dim={expected_dim}，实际 {raw_features.shape}"
            )
        labels = []
        for raw_label in truth_path.read_text(encoding="utf-8").splitlines():
            label = raw_label.strip()
            if not label:
                continue
            if label not in action_to_id:
                raise ValueError(f"{name}: 未登记动作标签 {label!r}")
            labels.append(action_id_remap[action_to_id[label]])
        common = min(len(sequence), len(labels))
        if max_frames is not None:
            common = min(common, max_frames)
        if window is not None and common < window:
            print(f"  [skip] {name}: 采样帧 {common} < window {window}")
            continue
        if common <= 0:
            continue
        features.append(np.asarray(sequence[:common], dtype=np.float32))
        truths.append(np.asarray(labels[:common], dtype=np.int64))
        accepted_names.append(name)

    if not features:
        raise ValueError(f"{root} 的 legacy split={split!r} 没有可用序列")
    return features, truths, id2name, accepted_names


def load_split(
    data_cfg: dict,
    split: str,
    window: int | None = None,
    feature_schema: dict | None = None,
    *,
    max_videos: int | None = None,
    max_frames: int | None = None,
    video_limit: int | None = None,
):
    """读某个 split 目录的全部视频，返回 (features_list, truths_list, id2name)。

    features_list[i] 形如 ``[T_i, F]``（float32），truths_list[i] 形如 ``[T_i]``（int64），
    索引与 ``labels/<split>/`` 下的视频对齐（同 ``split_video_names`` 的顺序）。若给了
    ``window``，跳过 ``T < window`` 的过短序列（窗口喂入 SlidingWindowDataset 无法开窗）并告警。
    ``feature_schema.mask_targets`` 可按 ``frames/data.yaml`` 的目标名或 ID 遮罩整组特征块
    （bbox 契约 5 维 / ROI 契约 18 维）。
    ``max_videos`` / ``max_frames`` 仅用于显式 smoke 评测限制，训练调用不传这两个参数。
    ``video_limit`` 取清单前 k 个视频（训练子采样口径，见 :func:`resolve_train_video_limit`），
    与 ``split_video_names(video_limit=...)`` 严格同序同批。
    """
    feature_version = (feature_schema or {}).get("version", "actionmixed-bbox-8cls-v1")
    if feature_version == LEGACY_FEATURE_VERSION:
        features, truths, id2name, _names = _load_legacy_endo_split(
            data_cfg,
            split,
            window=window,
            feature_schema=feature_schema,
            max_videos=max_videos if video_limit is None else video_limit,
            max_frames=max_frames,
        )
        return features, truths, id2name

    root = Path(data_cfg["root"])
    frames_dir = root / data_cfg.get("frames_dir", "frames") / split
    source_id2name = load_action_mapping(root, data_cfg.get("action_mapping", "labels/data.yaml"))
    class_order = (feature_schema or {}).get("class_order")
    if class_order is None:
        id2name = source_id2name
        action_id_remap = {source_id: source_id for source_id in source_id2name}
    else:
        if not isinstance(class_order, list) or not all(isinstance(name, str) for name in class_order):
            raise ValueError("feature_schema.class_order 必须是类别名列表")
        if len(class_order) != len(set(class_order)):
            raise ValueError("feature_schema.class_order 不能包含重复类别")
        source_names = set(source_id2name.values())
        if set(class_order) != source_names:
            raise ValueError(
                "feature_schema.class_order 与数据集动作类别不一致: "
                f"configured={class_order}, dataset={list(source_id2name.values())}"
            )
        name_to_target = {name: index for index, name in enumerate(class_order)}
        action_id_remap = {
            source_id: name_to_target[name] for source_id, name in source_id2name.items()
        }
        id2name = {index: name for index, name in enumerate(class_order)}
    mask_target_ids = resolve_mask_target_ids(data_cfg, feature_schema)
    image_embed_recipe = is_image_embed_version(feature_version)
    embed_root = resolve_embedding_root(data_cfg) if image_embed_recipe else None
    embed_dim = resolve_embed_dim(feature_schema) if image_embed_recipe else 0
    roi_recipe = feature_version == ROI_FEATURE_VERSION
    roi_presence_recipe = feature_version == ROI_PRESENCE_VERSION
    roi_v2_recipe = feature_version == ROI_GRID_V2_VERSION
    hand_recipe = feature_version == HAND_BBOX_VERSION
    global_hand_recipe = feature_version == GLOBAL_HAND_BBOX_VERSION
    clean_recipe = feature_version in CLEAN_FEATURE_DIMS
    detection_mapping = load_detection_mapping(data_cfg) if clean_recipe else None
    fps = float(data_cfg.get("fps", 7.5))
    confidence_default = float(
        (feature_schema or {}).get("detection_confidence_default", 1.0)
    )

    features, truths = [], []
    for stem, frame_ids, action_ids in _iter_split_sequences(data_cfg, split, window,
                                                             limit=video_limit):
        if max_videos is not None and len(features) >= max_videos:
            break
        if max_frames is not None:
            frame_ids = frame_ids[:max_frames]
            action_ids = action_ids[:max_frames]
            if window is not None and len(frame_ids) < window:
                continue
        frame_paths = [frames_dir / f"{stem}-{frame_id:06d}.txt" for frame_id in frame_ids]
        if image_embed_recipe:
            bbox_feats = np.stack(
                [
                    featurize_frame_bbox(path, mask_target_ids=mask_target_ids)
                    for path in frame_paths
                ]
            ).astype(np.float32)  # [T, 40]
            embeddings = load_video_embeddings(
                embed_root,
                split,
                stem,
                embed_dim,
                # smoke 截断（max_frames）时标签行已被截短，此时只按前缀切片
                expected_rows=None if max_frames is not None else len(frame_ids),
            )
            embeddings = embeddings[: len(frame_ids)]  # [T, embed_dim]
            if embeddings.shape[0] != len(frame_ids):
                raise ValueError(
                    f"embedding 行数少于标签行数: {stem} {embeddings.shape[0]} < {len(frame_ids)}"
                )
            feats = np.concatenate([bbox_feats, embeddings], axis=1)  # [T, 40 + embed_dim]
        elif clean_recipe:
            feats, _feature_names, actual_version = build_clean_bbox_features(
                frame_paths,
                detection_mapping=detection_mapping or {},
                feature_version=feature_version,
                fps=fps,
                confidence_default=confidence_default,
                mask_target_ids=mask_target_ids,
            )
            if actual_version != feature_version:
                raise ValueError(
                    f"CLEAN feature recipe 返回版本 {actual_version!r}，期望 {feature_version!r}"
                )
        elif roi_recipe:
            feats = np.stack(
                [
                    build_roi_frame_features(path, mask_target_ids=mask_target_ids)
                    for path in frame_paths
                ]
            ).astype(np.float32)
        elif roi_presence_recipe:
            feats = np.stack(
                [
                    build_roi_presence_frame_features(path, mask_target_ids=mask_target_ids)
                    for path in frame_paths
                ]
            ).astype(np.float32)
        elif roi_v2_recipe:
            feats = np.stack(
                [
                    build_roi_grid_v2_frame_features(path, mask_target_ids=mask_target_ids)
                    for path in frame_paths
                ]
            ).astype(np.float32)
        elif hand_recipe:
            feats = np.stack(
                [
                    build_hand_frame_features(path, mask_target_ids=mask_target_ids)
                    for path in frame_paths
                ]
            ).astype(np.float32)
        elif global_hand_recipe:
            feats = np.stack(
                [
                    np.concatenate(
                        [
                            featurize_frame_bbox(path, mask_target_ids=mask_target_ids),
                            build_hand_frame_features(path, mask_target_ids=mask_target_ids),
                        ]
                    )
                    for path in frame_paths
                ]
            ).astype(np.float32)
        else:
            feats = np.stack(
                [
                    featurize_frame_bbox(path, mask_target_ids=mask_target_ids)
                    for path in frame_paths
                ]
            ).astype(np.float32)
        features.append(feats)
        truths.append(
            np.asarray([action_id_remap[action_id] for action_id in action_ids], dtype=np.int64)
        )

    if not features:
        raise ValueError(f"{root / data_cfg.get('labels_dir', 'labels') / split} 下没有可用序列（可能都短于 window={window}）")
    return features, truths, id2name


def split_video_names(
    data_cfg: dict,
    split: str,
    window: int | None = None,
    *,
    max_videos: int | None = None,
    max_frames: int | None = None,
    video_limit: int | None = None,
) -> list[str]:
    """与 ``load_split`` 完全一致顺序的视频名列表（供可视化把逐帧预测贴回具体视频）。"""

    root = Path(data_cfg["root"])
    if (root / "mapping.txt").is_file() and (root / "features").is_dir():
        _features, _truths, _mapping, names = _load_legacy_endo_split(
            data_cfg,
            split,
            window=window,
            feature_schema={
                "version": LEGACY_FEATURE_VERSION,
                "dim": data_cfg.get("input_dim", 20),
            },
            max_videos=max_videos if video_limit is None else video_limit,
            max_frames=max_frames,
        )
        return names
    names = []
    for stem, frame_ids, _action_ids in _iter_split_sequences(data_cfg, split, window,
                                                             limit=video_limit):
        if max_frames is not None and window is not None and len(frame_ids[:max_frames]) < window:
            continue
        names.append(stem)
        if max_videos is not None and len(names) >= max_videos:
            break
    return names


def resolve_train_video_fraction(data_cfg: dict) -> float | None:
    """读取并校验 ``data.train_video_fraction``（训练视频子采样比例）。

    缺省 ``None`` = 用整个 train split（历史行为）。合法区间 ``(0, 1]``；非法值立即报错，
    不静默退回全量——学习曲线的横轴口径不能靠猜。
    """

    raw = data_cfg.get("train_video_fraction")
    if raw is None:
        return None
    try:
        fraction = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"data.train_video_fraction 必须是 (0, 1] 内的数，实际 {raw!r}") from exc
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"data.train_video_fraction 必须在 (0, 1] 内，实际 {fraction}")
    return fraction


def resolve_train_video_limit(data_cfg: dict, split: str | None = None) -> int | None:
    """把比例换算成"取前 k 个视频"的 k；未设置时返回 ``None``（全量）。

    口径：k = max(1, round(视频数 × fraction))，按 manifest/目录顺序取前 k 个——同一配置、
    同一 split 下每次切的是同一批视频（可复现的学习曲线横轴）。比例 ≥1 或 round 后等于全量
    时返回 ``None``，即不做子采样。
    """

    fraction = resolve_train_video_fraction(data_cfg)
    if fraction is None:
        return None
    total = count_split_videos(data_cfg, split or data_cfg["split_train"])
    limit = max(1, int(round(total * fraction)))
    return None if limit >= total else limit


def build_dataset_provenance(data_cfg: dict, feature_schema: dict | None) -> dict:
    """构造 checkpoint 使用的数据集版本、revision、split fingerprint 和映射摘要。"""

    fraction = resolve_train_video_fraction(data_cfg)
    subsample = {"train_video_fraction": fraction} if fraction is not None else {}
    dataset_ref = data_cfg.get("dataset_ref")
    if not dataset_ref:
        return {
            "registered": False,
            "id": data_cfg.get("name") or str(data_cfg.get("root")),
            **subsample,
        }
    from ..core.catalog import (
        get_dataset_specs,
        get_dataset_split,
        manifest_fingerprint,
        read_split_items,
    )

    specs = get_dataset_specs(str(dataset_ref))
    baseline = specs[0]
    roles = {
        role: str(data_cfg[key])
        for role, key in (("train", "split_train"), ("val", "split_val"), ("eval", "split_eval"))
        if data_cfg.get(key)
    }
    splits: dict[str, dict] = {}
    for split in sorted(set(roles.values())):
        spec = get_dataset_split(str(dataset_ref), split)
        splits[split] = {
            "testset_id": spec.id,
            "fingerprint_sha256": manifest_fingerprint(spec),
            "num_items": len(read_split_items(spec)),
        }

    root = Path(data_cfg["root"])
    legacy = baseline.feature_mapping == LEGACY_FEATURE_VERSION
    action_mapping = (
        root / "mapping.txt"
        if legacy
        else root / data_cfg.get("action_mapping", "labels/data.yaml")
    )
    detection_mapping = (
        None
        if legacy
        else root / data_cfg.get("frames_dir", "frames") / "data.yaml"
    )

    def mapping_info(path: Path) -> dict:
        return {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    provenance = {
        "registered": True,
        "id": str(dataset_ref),
        "version": baseline.dataset_version,
        "revision": baseline.dataset_revision,
        "feature_mapping": baseline.feature_mapping,
        "input_dim": baseline.input_dim,
        "labels": list(baseline.labels),
        "split_overlap_policy": baseline.split_overlap_policy,
        "roles": roles,
        "splits": splits,
        "action_mapping": mapping_info(action_mapping),
        "feature_schema": dict(feature_schema or {}),
        # 训练视图：子采样比例（未设置时不写该键），供 checkpoint 自证"这版权重是在多少数据上训的"。
        **subsample,
    }
    if detection_mapping is not None:
        provenance["detection_mapping"] = mapping_info(detection_mapping)
    return provenance


def assert_resume_dataset_compatible(checkpoint_meta: dict, current: dict) -> None:
    """恢复训练时要求数据版本、train fingerprint 与训练视图完全一致，拒绝静默混训。"""

    previous = checkpoint_meta.get("dataset")
    # 训练子采样口径先于"是否登记"检查：换子集 = 换训练视图，与数据集是否登记无关。
    if isinstance(previous, dict) and (
        previous.get("train_video_fraction") != current.get("train_video_fraction")
    ):
        raise ValueError(
            "resume 数据集不兼容: train_video_fraction "
            f"checkpoint={previous.get('train_video_fraction')!r} current={current.get('train_video_fraction')!r}"
        )
    if not current.get("registered"):
        return
    if not isinstance(previous, dict) or not previous.get("registered"):
        raise ValueError("当前训练使用已登记数据集，但 resume checkpoint 缺少数据集溯源")
    for key in ("id", "version", "revision", "feature_mapping", "labels"):
        if previous.get(key) != current.get(key):
            raise ValueError(
                f"resume 数据集不兼容: {key} checkpoint={previous.get(key)!r} "
                f"current={current.get(key)!r}"
            )
    previous_train = (previous.get("splits") or {}).get((previous.get("roles") or {}).get("train"), {})
    current_train = (current.get("splits") or {}).get((current.get("roles") or {}).get("train"), {})
    if previous_train.get("fingerprint_sha256") != current_train.get("fingerprint_sha256"):
        raise ValueError("resume 数据集不兼容: train split fingerprint 已变化")


def build_temporal_meta(
    model_cfg: dict,
    feature_schema: dict,
    pipeline: str,
    window: int,
    num_params: int,
    train_cfg: dict,
    trained_at: str,
    augmentation: dict | None = None,
    dataset: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """构造 checkpoint 重建元信息（两条时序流水线共用口径）。

    ``type`` 取自 ``model_cfg["type"]``，是 checkpoint 自描述与兼容校验的硬性键
    （见 core.integrity.check_checkpoint_config）。``extra`` 供个别模型补充字段
    （如 MS-TCN 的 ``normalizer``）。
    """
    meta = {
        "type": model_cfg["type"],
        "input_dim": model_cfg["input_dim"],
        "num_classes": model_cfg["num_classes"],
        "model": model_cfg,
        "feature_schema": feature_schema,
        "pipeline": pipeline,
        "window": window,
        "num_params": num_params,
        "trained_at": trained_at,
        "train": train_cfg,
    }
    if augmentation:
        meta["augmentation"] = augmentation
    if dataset:
        meta["dataset"] = dataset
    if extra:
        meta.update(extra)
    return meta


def build_external_temporal_meta(cfg: dict, pipeline: str) -> dict:
    """由 exploratory YAML 构造未绑定外部时序权重的临时重建信息。

    该信息只存在于本次评测内存和预测事实中，不写成可信 sidecar；权重仍须通过
    ``model.load_state_dict(..., strict=True)`` 的完整形状与参数键校验。
    """

    model_cfg = dict(cfg["model"])
    meta = {
        "type": model_cfg["type"],
        "pipeline": pipeline,
        "input_dim": model_cfg["input_dim"],
        "num_classes": model_cfg["num_classes"],
        "model": model_cfg,
        "feature_schema": dict(cfg.get("feature_schema") or {}),
        "num_params": None,
        "source": "missing_meta_fallback",
    }
    window = (cfg.get("train") or {}).get("window")
    if window is not None:
        meta["window"] = int(window)
    return meta


def resolve_external_temporal_meta(cfg: dict, pipeline: str) -> dict | None:
    """仅在 exploratory 且显式打开 ``allow_missing_meta`` 时返回 YAML fallback。"""

    mode = (cfg.get("evaluation") or {}).get("mode", "formal")
    allow_missing = bool((cfg.get("model") or {}).get("allow_missing_meta", False))
    if mode != "exploratory" or not allow_missing:
        return None
    return build_external_temporal_meta(cfg, pipeline)
