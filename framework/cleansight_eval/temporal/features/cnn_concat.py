"""S2/S3 契约：bbox/手部几何特征 ⊕ 全帧 CNN embedding 的 PCA 投影拼接。

见 docs/FEATURE_SCHEME_S2_S3_SPEC.md §2。embedding 由
``temporal/features/extract_embeddings.py`` 离线预计算（resnet18, 512 维，
与标签行一一对齐），PCA-80 在 train split 上拟合后存 `pca80.npz`
（键：mean[512], components[80,512]）。投影口径：z = (e-mean)@components.T，
clip 到 [-5,5] 后线性映射到 [0,1]——因果（val 只用 train 拟合的固定变换）、
确定性、无状态。

- `actionmixed-bbox-cnn-resnet18-v1`：bbox-40 ⊕ emb-80 = **120 维**（blocks=3）
- `actionmixed-bbox-hand-cnn-v1`：bbox-40 ⊕ hand-40 ⊕ emb-80 = **160 维**（blocks=4）

npy 或 pca80.npz 缺失时显式报错，不静默回退纯 bbox。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CNN_BBOX_VERSION = "actionmixed-bbox-cnn-resnet18-v1"
CNN_HAND_VERSION = "actionmixed-bbox-hand-cnn-v1"
CNN_FEATURE_DIMS = {
    CNN_BBOX_VERSION: 120,
    CNN_HAND_VERSION: 160,
}
EMBED_DIM_RAW = 512
EMBED_DIM_PROJ = 80
CLIP_RANGE = 5.0


def load_pca(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    for key in ("mean", "components"):
        if key not in data:
            raise ValueError(f"pca80.npz 缺少键 {key!r}: {npz_path}")
    comps = data["components"]
    if comps.shape != (EMBED_DIM_PROJ, EMBED_DIM_RAW):
        raise ValueError(
            f"PCA components 形状 {comps.shape} 与 ({EMBED_DIM_PROJ}, {EMBED_DIM_RAW}) 不符"
        )
    return {"mean": data["mean"].astype(np.float32), "components": comps.astype(np.float32)}


def project_embedding(emb_vec: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    z = (emb_vec.astype(np.float32) - pca["mean"]) @ pca["components"].T
    return (np.clip(z, -CLIP_RANGE, CLIP_RANGE) + CLIP_RANGE) / (2.0 * CLIP_RANGE)


def build_cnn_concat_frame(
    bbox_vec: np.ndarray,
    hand_vec: np.ndarray | None,
    emb_vec: np.ndarray | None,
    pca: dict[str, np.ndarray],
    *,
    include_hand: bool,
) -> np.ndarray:
    if emb_vec is None:
        raise ValueError("embedding 缺帧：S2/S3 契约要求逐帧 embedding（缺帧应预计算为全零行）")
    parts = [bbox_vec]
    if include_hand:
        if hand_vec is None:
            raise ValueError("hand 契约需要 hand 40 维向量")
        parts.append(hand_vec)
    parts.append(project_embedding(emb_vec, pca))
    return np.concatenate(parts).astype(np.float32)
