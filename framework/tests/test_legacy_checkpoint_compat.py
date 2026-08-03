"""历史时序 checkpoint 迁入 framework 后的严格加载与因果性测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from framework.cleansight_eval.temporal.models import build_model, is_causal


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("model_cfg", "checkpoint"),
    [
        (
            {
                "type": "legacy_gru_v1",
                "input_dim": 20,
                "num_classes": 3,
                "hidden": 128,
                "num_layers": 3,
            },
            ROOT / "registry/temporal/gru-v1/gru-final-20260704-150629.pt",
        ),
        (
            {
                "type": "legacy_causal_tcn_v1",
                "input_dim": 20,
                "num_classes": 3,
                "hidden_dims": [64, 64, 64],
            },
            ROOT / "registry/temporal/causal-tcn-v1/tcn-final-20260704-160652.pt",
        ),
        (
            {
                "type": "legacy_causal_transformer_v1",
                "input_dim": 20,
                "num_classes": 3,
                "d_model": 128,
                "nhead": 4,
                "num_layers": 3,
                "dim_feedforward": 256,
            },
            ROOT
            / "registry/temporal/causal-transformer-v1/transformer-final-20260704-161653.pt",
        ),
    ],
)
def test_registered_legacy_checkpoint_loads_strictly_and_is_causal(
    model_cfg: dict,
    checkpoint: Path,
) -> None:
    """真实历史权重必须严格加载，增加未来帧不能改变已有时间步输出。"""

    model = build_model(model_cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    torch.manual_seed(7)
    prefix = torch.randn(1, 8, 20)
    extended = torch.cat([prefix, torch.randn(1, 4, 20)], dim=1)
    with torch.no_grad():
        prefix_logits = model(prefix)
        extended_logits = model(extended)[:, : prefix.shape[1]]

    assert prefix_logits.shape == (1, 8, 3)
    assert torch.allclose(prefix_logits, extended_logits, atol=1e-5, rtol=1e-5)
    assert is_causal(model_cfg["type"]) is True
