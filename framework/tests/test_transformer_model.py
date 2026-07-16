"""Transformer 注册和输入输出契约测试。"""

import torch

from cleansight_eval.temporal.models import build_model, is_causal


def test_transformer_registry_forward_shape():
    model = build_model(
        {
            "type": "transformer",
            "input_dim": 40,
            "num_classes": 6,
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "max_len": 32,
        }
    )
    logits = model(torch.randn(2, 8, 40))
    assert logits.shape == (2, 8, 6)
    assert is_causal("transformer") is False
