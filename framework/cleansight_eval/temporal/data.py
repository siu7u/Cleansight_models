"""时序数据与 checkpoint 元信息（两条时序流水线共用）。

按 **features 契约**（bbox→40 维）把原始 cleansight-ActionMixed 数据读成逐帧特征序列。
本模块只负责"读原始数据 + 按 features 契约特征化"；加窗/整段等样本构造由各流水线自持
（见 ``full_sequence_pipeline`` / ``sliding_window_pipeline``）。另提供 checkpoint 重建
元信息的构造助手 ``build_temporal_meta``，两条时序流水线复用同一份口径。

真实数据格式（已按 train/val/test 目录切分）：

    <root>/labels/data.yaml                      动作类别（nc / names，6 类）
    <root>/labels/<split>/<video>.mp4.txt        每行 "frame_id action_id"（抽样帧，稀疏）
    <root>/frames/<split>/<video>.mp4-<f:06d>.txt 逐帧 YOLO bbox "class cx cy w h"（8 类）

特征口径（features 契约，version=actionmixed-bbox-8cls-v1）：按 8 类顺序，每类取该帧内
**面积最大的一个框** 编码 ``[presence, cx, cy, w, h]``，缺席则全零；拼成 8×5=40 维。
空 bbox 文件（无检测）→ 全零 40 维。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

N_DET_CLASSES = 8  # frames/data.yaml 的检测类数（每类 5 维）
FEATURE_DIM = N_DET_CLASSES * 5  # = 40


def featurize_frame_bbox(txt_path: Path, n_classes: int = N_DET_CLASSES) -> np.ndarray:
    """一帧 bbox → 定长特征向量（每类取面积最大框的 [presence, cx, cy, w, h]）。"""
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


def load_split(data_cfg: dict, split: str, window: int | None = None):
    """读某个 split 目录的全部视频，返回 (features_list, truths_list, id2name)。

    features_list[i] 形如 ``[T_i, 40]``（float32），truths_list[i] 形如 ``[T_i]``（int64），
    索引与 ``labels/<split>/`` 下的视频对齐。若给了 ``window``，跳过 ``T < window`` 的
    过短序列（窗口喂入 SlidingWindowDataset 无法开窗）并告警。
    """
    root = Path(data_cfg["root"])
    labels_dir = root / data_cfg.get("labels_dir", "labels") / split
    frames_dir = root / data_cfg.get("frames_dir", "frames") / split
    id2name = load_action_mapping(root, data_cfg.get("action_mapping", "labels/data.yaml"))

    if not labels_dir.is_dir():
        raise SystemExit(f"labels split 目录不存在: {labels_dir}")

    features, truths = [], []
    for label_file in sorted(labels_dir.glob("*.txt")):
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

        feats = np.stack(
            [featurize_frame_bbox(frames_dir / f"{stem}-{fid:06d}.txt") for fid in frame_ids]
        ).astype(np.float32)  # [T, 40]
        features.append(feats)
        truths.append(np.asarray(action_ids, dtype=np.int64))

    if not features:
        raise SystemExit(f"{labels_dir} 下没有可用序列（可能都短于 window={window}）")
    return features, truths, id2name


def build_temporal_meta(
    model_cfg: dict,
    feature_schema: dict,
    pipeline: str,
    window: int,
    num_params: int,
    train_cfg: dict,
    trained_at: str,
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
    if extra:
        meta.update(extra)
    return meta
