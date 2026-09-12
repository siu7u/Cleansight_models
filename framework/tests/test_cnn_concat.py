"""cnn_concat S2/S3 契约单测：投影口径、拼接维度、缺帧显式报错。"""

from __future__ import annotations

import numpy as np
import pytest

from cleansight_eval.temporal.features.cnn_concat import (
    CNN_BBOX_VERSION,
    CNN_FEATURE_DIMS,
    CNN_HAND_VERSION,
    EMBED_DIM_PROJ,
    EMBED_DIM_RAW,
    build_cnn_concat_frame,
    load_pca,
    project_embedding,
)


def _fake_pca() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    comps, _ = np.linalg.qr(rng.standard_normal((EMBED_DIM_RAW, EMBED_DIM_PROJ)))
    return {
        "mean": rng.standard_normal(EMBED_DIM_RAW).astype(np.float32),
        "components": comps.T.astype(np.float32),
    }


def test_project_embedding_is_bounded_and_deterministic():
    pca = _fake_pca()
    e = rng = np.random.default_rng(1).standard_normal(EMBED_DIM_RAW).astype(np.float32)
    p1 = project_embedding(e, pca)
    p2 = project_embedding(e, pca)
    assert p1.shape == (EMBED_DIM_PROJ,)
    assert 0.0 <= p1.min() and p1.max() <= 1.0
    np.testing.assert_array_equal(p1, p2)


def test_s2_concat_shape_and_layout():
    pca = _fake_pca()
    bbox = np.arange(40, dtype=np.float32)
    emb = np.zeros(EMBED_DIM_RAW, dtype=np.float32)
    frame = build_cnn_concat_frame(bbox, None, emb, pca, include_hand=False)
    assert frame.shape == (CNN_FEATURE_DIMS[CNN_BBOX_VERSION],)
    np.testing.assert_array_equal(frame[:40], bbox)  # 左半 = 全局 bbox 原值


def test_s3_concat_shape_and_layout():
    pca = _fake_pca()
    bbox = np.ones(40, dtype=np.float32)
    hand = np.full(40, 2.0, dtype=np.float32)
    emb = np.zeros(EMBED_DIM_RAW, dtype=np.float32)
    frame = build_cnn_concat_frame(bbox, hand, emb, pca, include_hand=True)
    assert frame.shape == (CNN_FEATURE_DIMS[CNN_HAND_VERSION],)
    np.testing.assert_array_equal(frame[:40], bbox)
    np.testing.assert_array_equal(frame[40:80], hand)


def test_missing_embedding_raises():
    pca = _fake_pca()
    with pytest.raises(ValueError, match="embedding 缺帧"):
        build_cnn_concat_frame(np.zeros(40, np.float32), None, None, pca, include_hand=False)


def test_load_pca_validates_shape(tmp_path):
    bad = tmp_path / "pca80.npz"
    np.savez(bad, mean=np.zeros(EMBED_DIM_RAW), components=np.zeros((3, EMBED_DIM_RAW)))
    with pytest.raises(ValueError, match="PCA components"):
        load_pca(bad)
