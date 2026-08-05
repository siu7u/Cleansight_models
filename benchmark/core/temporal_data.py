"""按统一 testset manifest 加载 `[T,F]` 时序特征和逐帧标签。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from framework.cleansight_eval.core.catalog import TestsetSpec, read_split_items, resolve_path


@dataclass(frozen=True)
class TemporalItem:
    """一段独立视频序列；视频边界在评估中不得被拼接。"""

    name: str
    features: np.ndarray
    labels: np.ndarray


def load_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """读取 `mapping.txt` 并校验类别编号唯一。"""

    action_to_index: dict[str, int] = {}
    index_to_action: dict[int, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        index_text, action = raw.split(maxsplit=1)
        index = int(index_text)
        if action in action_to_index or index in index_to_action:
            raise ValueError(f"mapping 含重复标签或编号: {raw}")
        action_to_index[action] = index
        index_to_action[index] = action
    return action_to_index, index_to_action


def _load_features(path: Path, input_dim: int, name: str) -> np.ndarray:
    """加载特征并依据钉定 input_dim 规范成 `[T,F]`。"""

    features = np.load(path)
    if features.ndim != 2:
        raise ValueError(f"{name}: 特征必须是二维，收到 {features.shape}")
    if features.shape[1] == input_dim:
        normalized = features
    elif features.shape[0] == input_dim:
        normalized = features.T
    else:
        raise ValueError(
            f"{name}: 无法对齐 input_dim={input_dim}，原始 shape={features.shape}"
        )
    return normalized.astype(np.float32, copy=False)


def _load_truth(path: Path, mapping: dict[str, int], name: str) -> np.ndarray:
    """把逐行动作名转换为 `[T]` 类别编号。"""

    labels = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        action = raw.strip()
        if not action:
            continue
        if action not in mapping:
            raise ValueError(f"{name}: 出现 mapping 未登记标签 {action!r}")
        labels.append(mapping[action])
    return np.asarray(labels, dtype=np.int64)


def load_temporal_items(
    spec: TestsetSpec,
    *,
    data_root: str | Path | None = None,
    max_videos: int | None = None,
    max_frames: int | None = None,
) -> tuple[list[TemporalItem], dict[int, str]]:
    """严格按 manifest 样本顺序加载序列，并在单视频内部对齐长度。"""

    if spec.family != "temporal":
        raise ValueError(f"需要 temporal testset，收到 {spec.family}")
    if spec.input_dim is None:
        raise ValueError(f"{spec.id} 缺少 input_dim")
    if max_videos is not None and max_videos <= 0:
        raise ValueError("max_videos 必须大于 0")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames 必须大于 0")

    root = (
        Path(data_root).expanduser().resolve()
        if data_root is not None
        else resolve_path(spec.data_root or "", spec.root)
    )
    mapping, index_to_action = load_mapping(root / "mapping.txt")
    expected_labels = tuple(index_to_action[index] for index in sorted(index_to_action))
    if expected_labels != spec.labels:
        raise ValueError(
            f"{spec.id}: mapping labels={expected_labels} 与 manifest={spec.labels} 不一致"
        )

    names = read_split_items(spec)
    if max_videos is not None:
        names = names[:max_videos]
    items = []
    for name in names:
        features = _load_features(root / "features" / f"{name}.npy", spec.input_dim, name)
        labels = _load_truth(root / "groundTruth" / f"{name}.txt", mapping, name)
        common = min(len(features), len(labels))
        if max_frames is not None:
            common = min(common, max_frames)
        if common <= 0:
            raise ValueError(f"{name}: 没有可评估帧")
        items.append(
            TemporalItem(name=name, features=features[:common], labels=labels[:common])
        )
    return items, index_to_action
