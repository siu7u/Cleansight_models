"""序列级数据增强（feature_jitter / temporal_scale）的契约测试。

覆盖三件事：**缺省逐位等同历史行为**、**同 seed 可复现**、
**语义不被破坏**（零仍是零、标签与特征同步重采样、滑窗禁用换长度）。
"""

from __future__ import annotations

import numpy as np
import pytest

from framework.cleansight_eval.temporal.data import (
    apply_sequence_augmentation,
    resolve_sequence_augmentation,
)

DATA_CFG = {"root": "/tmp/nonexistent", "split_train": "train", "split_eval": "test"}


def _sample(n_frames: int = 20, dim: int = 8) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(0)
    feats = [rng.random((n_frames, dim), dtype=np.float32)]
    feats[0][::3] = 0.0  # 制造"缺席"位置，用于验证零不被抖动
    labels = [rng.integers(0, 3, size=n_frames).astype(np.int64)]
    return feats, labels


def test_disabled_returns_inputs_untouched():
    feats, labels = _sample()
    out_f, out_l = apply_sequence_augmentation(feats, labels, DATA_CFG, None, seed=42)
    assert out_f is feats and out_l is labels  # 未启用时不复制、原对象返回


def test_unknown_augmentation_key_raises():
    with pytest.raises(ValueError, match="未知字段"):
        resolve_sequence_augmentation(DATA_CFG, {"grid_flip": {"enabled": True}})


def test_target_mask_key_is_accepted_by_sequence_resolver():
    # 两个解析器共用同一套顶层键，target_mask 单独出现时序列级增强应视为未启用
    spec = resolve_sequence_augmentation(DATA_CFG, {"target_mask": {"probability": 0.5, "targets": [0]}})
    assert spec is None


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"feature_jitter": {"sigma": -0.1}}, "sigma"),
        ({"feature_jitter": {"sigma": float("inf")}}, "sigma"),
        ({"feature_jitter": {"enabled": "yes"}}, "enabled"),
        ({"feature_jitter": {"sigma": 0.1, "mode": "gauss"}}, "未知字段"),
        ({"temporal_scale": {"max_rate": 1.0}}, "max_rate"),
        ({"temporal_scale": {"max_rate": -0.1}}, "max_rate"),
    ],
)
def test_invalid_config_raises(payload, match):
    with pytest.raises(ValueError, match=match):
        resolve_sequence_augmentation(DATA_CFG, payload)


def test_jitter_keeps_zeros_and_is_reproducible():
    feats, labels = _sample()
    cfg = {"feature_jitter": {"sigma": 0.2}}
    a, _ = apply_sequence_augmentation(feats, labels, DATA_CFG, cfg, seed=7)
    b, _ = apply_sequence_augmentation(feats, labels, DATA_CFG, cfg, seed=7)
    c, _ = apply_sequence_augmentation(feats, labels, DATA_CFG, cfg, seed=8)
    assert np.array_equal(a[0], b[0])           # 同 seed 逐位可复现
    assert not np.array_equal(a[0], c[0])       # 换 seed 结果不同
    zero_mask = feats[0] == 0.0
    assert np.array_equal(a[0][zero_mask], np.zeros(int(zero_mask.sum()), dtype=np.float32))
    assert (a[0][~zero_mask] >= 0.0).all()      # 抖动不产生负计数
    assert a[0].shape == feats[0].shape         # jitter 不改形状


def test_temporal_scale_resamples_features_and_labels_together():
    feats, labels = _sample(n_frames=40)
    cfg = {"temporal_scale": {"max_rate": 0.25}}
    out_f, out_l = apply_sequence_augmentation(feats, labels, DATA_CFG, cfg, seed=3)
    assert out_f[0].shape[0] == out_l[0].shape[0]          # 特征/标签同步
    assert out_f[0].shape[1] == feats[0].shape[1]          # 维度不变
    assert abs(out_f[0].shape[0] - 40) / 40 <= 0.26        # 倍率落在标称范围内
    assert out_f[0].dtype == np.float32


def test_temporal_scale_disabled_for_fixed_window_pipeline():
    feats, labels = _sample()
    cfg = {"temporal_scale": {"max_rate": 0.2}}
    with pytest.raises(ValueError, match="滑窗"):
        apply_sequence_augmentation(
            feats, labels, DATA_CFG, cfg, seed=1, allow_temporal_scale=False
        )


def test_jitter_still_allowed_for_fixed_window_pipeline():
    feats, labels = _sample()
    out_f, _ = apply_sequence_augmentation(
        feats, labels, DATA_CFG, {"feature_jitter": {"sigma": 0.1}}, seed=1,
        allow_temporal_scale=False,
    )
    assert out_f[0].shape == feats[0].shape


def test_both_augmentations_compose():
    feats, labels = _sample(n_frames=30)
    cfg = {"feature_jitter": {"sigma": 0.1}, "temporal_scale": {"max_rate": 0.2}}
    out_f, out_l = apply_sequence_augmentation(feats, labels, DATA_CFG, cfg, seed=11)
    assert out_f[0].shape[0] == out_l[0].shape[0]
    assert out_f[0].shape[1] == feats[0].shape[1]


def test_mismatched_lengths_raise():
    feats, _ = _sample()
    with pytest.raises(ValueError, match="条数不一致"):
        apply_sequence_augmentation(
            feats, [], DATA_CFG, {"feature_jitter": {"sigma": 0.1}}, seed=1
        )
