"""形态 B（像素特征进时序）契约：bbox 块 + 逐帧整图 CNN embedding 块拼接。

契约布局（``actionmixed-bbox-embed-mbv3s-v1``，见 docs/features/IMAGE_FEATURE_TRAINING.md §4.1）：

    [ 40 维 bbox 块 | 576 维图像块 ]  = 616 维

- **bbox 块**：逐帧读 ``frames/<split>/<video>-<f:06d>.txt``，语义与 ``actionmixed-bbox-8cls-v1``
  完全一致（8 检测类 × ``[presence, cx, cy, w, h]``，每类取面积最大框）；``mask_targets``
  只作用于这一块（图像块没有检测类结构，无法按类遮罩）；
- **图像块**：读 ``extract_embeddings.py`` 的离线产物 ``<embed_root>/<split>/<video>.mp4.npy``
  （``[T, feat_dim]``）。backbone 冻结、预计算，训练侧只读 npy，不依赖图像与 GPU；
- **对齐**：按标签行**位置**对齐（第 i 个标签帧 ↔ npy 第 i 行）；行数/列数不符立即报错，
  不静默错位——陈旧产物错位比训练失败更贵；
- **因果、无状态**：每帧只依赖当前帧图像与当前帧检测框；缺图帧由提取器补零并记录在
  ``meta.json``（``missing_images``）。

``embed_root`` 来自配置 ``data.embedding_root``（catalog 登记的 ``feature_embed.root`` 为
数据集合约的默认值）：换 backbone 或换 embedding 数据源时只改登记/覆盖该字段，不改代码；
embedding 维度变化则必须新增 feature mapping 版本（modelset-quality Feature Mapping Rules）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_EMBED_VERSION = "actionmixed-bbox-embed-mbv3s-v1"
# 同族契约的版本前缀：core/catalog 按前缀校验维度声明，core 层不 import 任何 recipe。
IMAGE_EMBED_VERSION_PREFIX = "actionmixed-bbox-embed-"
EMBED_BBOX_DIM = 40  # 8 检测类 × 5 维（actionmixed-bbox-8cls-v1）
EMBED_FEAT_DIM = 576  # mobilenet_v3_small 整图 embedding 维度（extract_embeddings 产物）
IMAGE_EMBED_DIM = EMBED_BBOX_DIM + EMBED_FEAT_DIM  # = 616


def is_image_embed_version(version: str | None) -> bool:
    """``feature_schema.version`` 是否属于「bbox + 图像 embedding」拼接契约。"""

    return isinstance(version, str) and version.startswith(IMAGE_EMBED_VERSION_PREFIX)


def resolve_embed_dim(feature_schema: dict | None) -> int:
    """契约声明的图像块宽度 = ``dim - EMBED_BBOX_DIM``；非法维度直接报错。

    用于校验 npy 产物的列数与模型投影头宽度是否与契约一致。
    """

    dim = (feature_schema or {}).get("dim")
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= EMBED_BBOX_DIM:
        raise ValueError(
            f"图像 embedding 契约的 feature_schema.dim 必须是 > {EMBED_BBOX_DIM} 的整数，"
            f"实际 {dim!r}"
        )
    return dim - EMBED_BBOX_DIM


def resolve_embedding_root(data_cfg: dict) -> Path:
    """取 embedding 产物根目录（catalog 的 ``feature_embed.root`` 注入 ``data.embedding_root``）。"""

    raw = data_cfg.get("embedding_root")
    if not raw:
        raise ValueError(
            "图像 embedding 契约需要 data.embedding_root（catalog 数据集条目声明 "
            "feature_embed.root，或显式覆盖为绝对路径）"
        )
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"embedding 产物根目录不存在: {root}")
    return root


def load_video_embeddings(
    embed_root: Path,
    split: str,
    stem: str,
    feat_dim: int,
    expected_rows: int | None = None,
) -> np.ndarray:
    """读一个视频的逐帧 embedding ``[T, feat_dim]``（float32）。

    ``stem`` 为视频名（含 ``.mp4``），与 ``extract_embeddings.py`` 的产物命名一致。
    ``expected_rows`` 给出该视频的标签行数时要求严格相等（产物陈旧即报错）；``max_frames``
    之类的 smoke 截断路径传 None，由调用方按前缀切片。缺文件、维度不符、行数不符都直接抛错。
    """

    path = Path(embed_root) / split / f"{stem}.npy"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 embedding 产物: {path}（先跑 features/extract_embeddings.py 生成）"
        )
    array = np.load(path)
    if array.ndim != 2 or array.shape[1] != feat_dim:
        raise ValueError(
            f"embedding 产物维度与契约不符: {path} shape={array.shape}，契约要求 [T, {feat_dim}]"
        )
    if expected_rows is not None and array.shape[0] != expected_rows:
        raise ValueError(
            f"embedding 行数与标签行数不一致: {path} {array.shape[0]} != {expected_rows}"
            "（产物与当前标签不同源，请重跑 extract_embeddings）"
        )
    return array.astype(np.float32, copy=False)
