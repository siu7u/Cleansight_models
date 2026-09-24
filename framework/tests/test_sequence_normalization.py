"""序列级归一化（model.sequence_normalization = none | demean）的口径与边界。

背景：第十三轮探针发现"按视频去均值"能把最差类 flush 的 train→test 可迁移性从 0.281 修到
0.604，但那是探针读数；本轮把它实现在模型里（全序列 mstcn/mstcn2），因此需要测试钉住：
① demean 真的是序列级（对整段电平平移不变）；② 口径字符串自描述；③ 不支持的模型必须显式报错
（不能"配了却没生效"）。
"""

from __future__ import annotations

import pytest
import torch

from framework.cleansight_eval.temporal.models import build_model

CFG = {"input_dim": 8, "num_classes": 3, "hidden": 8}


def _logits(model, x):
    with torch.no_grad():
        return model(x)


def test_demean_is_sequence_level_and_none_is_not():
    x = torch.randn(2, 40, 8) * 2.0 + 5.0        # 带明显电平
    shifted = x + 3.0                            # 整段平移
    none_model = build_model({"type": "mstcn", **CFG, "sequence_normalization": "none"}).eval()
    demean_model = build_model({"type": "mstcn", **CFG, "sequence_normalization": "demean"}).eval()

    delta_none = float((_logits(none_model, x) - _logits(none_model, shifted)).abs().max())
    delta_demean = float((_logits(demean_model, x) - _logits(demean_model, shifted)).abs().max())
    assert delta_none > 1e-3, "none 口径不应具备电平平移不变性"
    assert delta_demean < 1e-5, "demean 口径应对整段电平平移不变"


def test_demean_supported_on_mstcn2_and_declared_in_spec():
    none_spec = build_model({"type": "mstcn2", **CFG}).normalizer_spec()
    demean_spec = build_model({"type": "mstcn2", **CFG, "sequence_normalization": "demean"}).normalizer_spec()
    assert none_spec == "zscore/train-set/buffers/v1"          # 旧 checkpoint 口径不变
    assert demean_spec.endswith("+sequence-demean/v1") and demean_spec.startswith(none_spec.split("+")[0])


@pytest.mark.parametrize("model_type", ["gru", "transformer", "clean_mstcn_bilstm"])
def test_unsupported_models_reject_demean(model_type):
    """因果/滑窗模型看不到整段序列，必须显式拒绝而不是静默忽略。"""

    with pytest.raises(ValueError, match="sequence_normalization"):
        build_model({"type": model_type, **CFG, "sequence_normalization": "demean"})


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="sequence_normalization"):
        build_model({"type": "mstcn", **CFG, "sequence_normalization": "zscore"})
