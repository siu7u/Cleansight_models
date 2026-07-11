"""checkpoint 携带重建元信息且拒绝错配加载（需求 §7.2 / §8.1）。"""

import pytest
import torch

from cleansight_eval.core.checkpoint import load_checkpoint, save_checkpoint
from cleansight_eval.core.integrity import CompatibilityError
from cleansight_eval.temporal.family import get_family


def _make_ckpt(tmp_path):
    family = get_family("gru")
    cfg = {"input_dim": 20, "num_classes": 3, "hidden": 16, "num_layers": 1}
    model = family.build_network(cfg)
    meta = family.checkpoint_meta(cfg, {"dim": 20, "version": "legacy-20d-v1"}, extra={"window": 64})
    path = tmp_path / "gru.pt"
    save_checkpoint(path, model.state_dict(), meta)
    return path, cfg


def test_roundtrip_and_meta(tmp_path):
    path, cfg = _make_ckpt(tmp_path)
    state, meta = load_checkpoint(path, expected={"family": "gru", "input_dim": 20, "num_classes": 3})
    assert meta["family"] == "gru"
    assert meta["model"] == cfg
    assert "rnn.weight_ih_l0" in state


def test_wrong_input_dim_rejected(tmp_path):
    path, _ = _make_ckpt(tmp_path)
    with pytest.raises(CompatibilityError):
        load_checkpoint(path, expected={"family": "gru", "input_dim": 64, "num_classes": 3})


def test_wrong_family_rejected(tmp_path):
    path, _ = _make_ckpt(tmp_path)
    with pytest.raises(CompatibilityError):
        load_checkpoint(path, expected={"family": "transformer", "input_dim": 20, "num_classes": 3})


def test_missing_meta_rejected(tmp_path):
    path, _ = _make_ckpt(tmp_path)
    (tmp_path / "gru.pt.meta.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(path, expected=None)
