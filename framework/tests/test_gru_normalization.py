"""GRU 输入归一化（P2a，``model.normalization``）的模型层测试。

覆盖：默认直通（不注册 buffer、provenance 为 None）、z-score 统计与守卫、归一化等价性
（对输入做同样变换应得到同样输出）、截断、工厂接线与非法值报错。
"""

import numpy as np
import pytest
import torch

from cleansight_eval.core.config import validate_config
from cleansight_eval.temporal.models import build_model
from cleansight_eval.temporal.models.gru import GRUClassifier
from cleansight_eval.temporal.models.mstcn import MSTCN


def _features(seed: int = 0, videos: int = 3, frames: int = 20, dim: int = 6):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(videos):
        x = rng.normal(loc=0.0, scale=1.0, size=(frames, dim)).astype(np.float32)
        x[:, -1] = 0.0  # 常量维：归一化守卫应把它置 std=1
        x[:, 0] = x[:, 0] * 0.05 + 0.3  # 小量纲维（模拟 max_area 通道）
        out.append(x)
    return out


def _gru(**kwargs) -> GRUClassifier:
    torch.manual_seed(0)
    return GRUClassifier(input_dim=6, num_classes=3, hidden=8, num_layers=1, **kwargs)


def test_default_none_is_passthrough():
    """默认不归一化：无 buffer、provenance 为 None、fit_normalization 为空操作。"""

    model = _gru()
    assert model.normalization_enabled is False
    assert model.normalizer_spec() is None
    assert not any(name.startswith("norm_") for name, _ in model.named_buffers())

    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.fit_normalization(_features())
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_zscore_writes_train_statistics_with_guard():
    """z-score 统计来自训练特征，常量维 std 守卫为 1.0。"""

    features = _features()
    x = np.concatenate(features, axis=0)
    model = _gru(normalization="zscore")
    model.fit_normalization(features)

    assert np.allclose(model.norm_mean.numpy(), x.mean(axis=0), atol=1e-5)
    expected_std = x.std(axis=0)
    expected_std[expected_std < 1e-3] = 1.0
    assert np.allclose(model.norm_std.numpy(), expected_std, atol=1e-5)
    assert float(model.norm_std[-1]) == 1.0  # 常量维守卫生效
    assert model.normalizer_spec() == "zscore/train-set/buffers/v1"


def test_zscore_forward_equals_manual_normalization():
    """开启归一化后，forward(x) 等价于"先按统计手工标准化再喂直通模型"。"""

    features = _features()
    x = np.concatenate(features, axis=0)[:5]
    normalized = _gru(normalization="zscore")
    normalized.fit_normalization(features)

    passthrough = _gru()
    passthrough.load_state_dict(normalized.state_dict(), strict=False)

    with torch.no_grad():
        got = normalized(torch.tensor(x).unsqueeze(0))
        mean, std = normalized.norm_mean.numpy(), normalized.norm_std.numpy()
        want = passthrough(torch.tensor((x - mean) / std).unsqueeze(0))

    assert torch.allclose(got, want, atol=1e-5)


def test_norm_clip_limits_standardized_values():
    """norm_clip 把标准化结果截断到 ±norm_clip（防止近零方差维被放大）。"""

    features = _features()
    x = np.concatenate(features, axis=0)[:5]
    clipped = _gru(normalization="zscore", norm_clip=1.0)
    clipped.fit_normalization(features)
    assert clipped.normalizer_spec() == "zscore/train-set/buffers/v1/clip=1"

    manual = _gru()
    manual.load_state_dict(clipped.state_dict(), strict=False)
    mean, std = clipped.norm_mean.numpy(), clipped.norm_std.numpy()
    z = np.clip((x - mean) / std, -1.0, 1.0)

    with torch.no_grad():
        assert torch.allclose(clipped(torch.tensor(x).unsqueeze(0)),
                              manual(torch.tensor(z).unsqueeze(0)), atol=1e-5)


def test_build_model_wires_normalization():
    """工厂把 model.normalization / norm_clip 透传给 GRU；MS-TCN 恒声明 z-score。"""

    cfg = {"type": "gru", "input_dim": 6, "num_classes": 3, "hidden": 8, "num_layers": 1,
           "normalization": "zscore", "norm_clip": 2}
    model = build_model(cfg)
    assert model.normalization_enabled is True
    assert model.normalizer_spec() == "zscore/train-set/buffers/v1/clip=2"

    assert build_model({"type": "gru", "input_dim": 6, "num_classes": 3}).normalizer_spec() is None
    assert MSTCN(in_dim=6, classes=3).normalizer_spec() == "zscore/train-set/buffers/v1"


def test_invalid_normalization_mode_rejected():
    with pytest.raises(ValueError, match="只支持"):
        _gru(normalization="minmax")


def test_config_whitelist_accepts_normalization_keys():
    """model.normalization / model.norm_clip 是合法配置键（白名单已登记）。"""

    validate_config({
        "schema_version": 1,
        "pipeline": "sliding_window_temporal",
        "model": {"type": "gru", "input_dim": 96, "num_classes": 6,
                  "normalization": "zscore", "norm_clip": 5},
        "data": {"dataset_ref": "temporal.actionmixed-auto-roi-v2"},
        "feature_schema": {"dim": 96, "version": "actionmixed-roi-grid-v2"},
    })
