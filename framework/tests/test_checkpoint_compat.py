"""checkpoint 携带重建元信息且拒绝错配加载（需求 §7.2 / §8.1）。"""

import numpy as np
import pytest
import torch

from cleansight_eval.core.checkpoint import (
    META_SCHEMA_VERSION,
    load_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
)
from cleansight_eval.core.integrity import CompatibilityError
from cleansight_eval.temporal.data import (
    assert_resume_dataset_compatible,
    build_temporal_meta,
    resolve_external_temporal_meta,
)
from cleansight_eval.temporal.full_sequence_pipeline import _load_eval_model
from cleansight_eval.temporal.models import build_model
from cleansight_eval.temporal.sliding_window_pipeline import SlidingWindowTemporalPipeline


def _make_ckpt(tmp_path):
    cfg = {"type": "gru", "input_dim": 20, "num_classes": 3, "hidden": 16, "num_layers": 1}
    model = build_model(cfg)
    meta = build_temporal_meta(
        cfg,
        {"dim": 20, "version": "legacy-20d-v1"},
        pipeline="sliding_window_temporal",
        window=64,
        num_params=sum(p.numel() for p in model.parameters()),
        train_cfg={"epochs": 1},
        trained_at="t0",
    )
    path = tmp_path / "gru.pt"
    save_checkpoint(path, model.state_dict(), meta)
    return path, cfg


def test_roundtrip_and_meta(tmp_path):
    path, cfg = _make_ckpt(tmp_path)
    state, meta = load_checkpoint(path, expected={"type": "gru", "input_dim": 20, "num_classes": 3})
    assert meta["type"] == "gru"
    assert meta["model"] == cfg
    assert meta["schema_version"] == META_SCHEMA_VERSION
    assert meta["checkpoint_binding"]["sha256"]
    assert meta["_metadata_integrity"]["bound"] is True
    assert "rnn.weight_ih_l0" in state


def test_wrong_input_dim_rejected(tmp_path):
    path, _ = _make_ckpt(tmp_path)
    with pytest.raises(CompatibilityError):
        load_checkpoint(path, expected={"type": "gru", "input_dim": 64, "num_classes": 3})


def test_wrong_type_rejected(tmp_path):
    path, _ = _make_ckpt(tmp_path)
    with pytest.raises(CompatibilityError):
        load_checkpoint(path, expected={"type": "transformer", "input_dim": 20, "num_classes": 3})


def test_missing_meta_rejected(tmp_path):
    path, _ = _make_ckpt(tmp_path)
    (tmp_path / "gru.pt.meta.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_checkpoint(path, expected=None)


def _external_cfg(*, allow_missing_meta=True, mode="exploratory", hidden=16):
    return {
        "pipeline": "sliding_window_temporal",
        "model": {
            "type": "gru",
            "input_dim": 20,
            "num_classes": 3,
            "hidden": hidden,
            "num_layers": 1,
            "allow_missing_meta": allow_missing_meta,
        },
        "data": {"name": "external-fixture", "split_eval": "test"},
        "feature_schema": {"dim": 20, "version": "legacy-20d-v1"},
        "evaluation": {"mode": mode, "measure_latency": False},
        "train": {"window": 2},
    }


def _write_external_state_dict(tmp_path, wrapper=None):
    cfg = _external_cfg()["model"]
    model = build_model(cfg)
    path = tmp_path / "external-gru.pt"
    state = model.state_dict()
    torch.save({wrapper: state} if wrapper else state, path)
    return path


def test_external_temporal_checkpoint_fallback_is_explicit_and_unbound(tmp_path):
    path = _write_external_state_dict(tmp_path)
    cfg = _external_cfg()
    fallback = resolve_external_temporal_meta(cfg, "sliding_window_temporal")

    state, meta = load_checkpoint(path, fallback_meta=fallback)

    assert "rnn.weight_ih_l0" in state
    assert meta["source"] == "missing_meta_fallback"
    assert meta["window"] == 2
    assert meta["_metadata_integrity"] == {
        "schema_version": 0,
        "bound": False,
        "source": "config_fallback",
    }


def test_external_temporal_checkpoint_accepts_common_state_dict_wrapper(tmp_path):
    path = _write_external_state_dict(tmp_path, wrapper="state_dict")
    fallback = resolve_external_temporal_meta(
        _external_cfg(), "sliding_window_temporal"
    )

    state, _meta = load_checkpoint(path, fallback_meta=fallback)

    assert "rnn.weight_ih_l0" in state


@pytest.mark.parametrize(
    ("mode", "allow_missing_meta"),
    [("exploratory", False), ("formal", True)],
)
def test_external_temporal_checkpoint_fallback_stays_closed_by_default(
    tmp_path, mode, allow_missing_meta
):
    path = _write_external_state_dict(tmp_path)
    cfg = _external_cfg(mode=mode, allow_missing_meta=allow_missing_meta)

    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            path,
            fallback_meta=resolve_external_temporal_meta(
                cfg, "sliding_window_temporal"
            ),
        )


def test_formal_checkpoint_loader_rejects_fallback_metadata(tmp_path):
    path = _write_external_state_dict(tmp_path)
    fallback = resolve_external_temporal_meta(
        _external_cfg(), "sliding_window_temporal"
    )

    with pytest.raises(ValueError, match="formal checkpoint"):
        load_checkpoint(
            path,
            require_meta_schema=True,
            fallback_meta=fallback,
        )


def test_external_full_sequence_checkpoint_keeps_strict_shape_check(tmp_path):
    path = _write_external_state_dict(tmp_path)
    cfg = _external_cfg(hidden=8)
    cfg["pipeline"] = "full_sequence_temporal"

    with pytest.raises(RuntimeError):
        _load_eval_model(cfg, str(path), torch.device("cpu"))


def test_external_sliding_checkpoint_runs_without_sidecar(tmp_path, monkeypatch):
    path = _write_external_state_dict(tmp_path)
    cfg = _external_cfg()
    import cleansight_eval.temporal.sliding_window_pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "load_split",
        lambda *_args, **_kwargs: (
            [np.zeros((3, 20), dtype=np.float32)],
            [np.zeros(3, dtype=np.int64)],
            {0: "idle", 1: "long", 2: "short"},
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "split_video_names",
        lambda *_args, **_kwargs: ["external-video"],
    )

    output = SlidingWindowTemporalPipeline().predict(
        cfg, str(path), torch.device("cpu")
    )

    assert output.metadata["checkpoint_metadata_source"] == "missing_meta_fallback"
    assert output.metadata["checkpoint_metadata_bound"] is False
    assert output.num_params > 0


def test_checkpoint_content_replacement_is_rejected(tmp_path):
    path, _ = _make_ckpt(tmp_path)
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="绑定摘要不一致"):
        load_checkpoint(path, expected=None)


def test_training_checkpoint_keeps_eval_loader_compatible(tmp_path):
    cfg = {"type": "gru", "input_dim": 20, "num_classes": 3, "hidden": 16, "num_layers": 1}
    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    meta = build_temporal_meta(
        cfg,
        {"dim": 20, "version": "legacy-20d-v1"},
        pipeline="sliding_window_temporal",
        window=64,
        num_params=sum(p.numel() for p in model.parameters()),
        train_cfg={"epochs": 2},
        trained_at="t0",
    )
    path = tmp_path / "last.pt"

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        meta=meta,
        best_metric={"name": "val_acc", "mode": "max", "value": 0.75, "epoch": 2},
    )

    payload, training_meta = load_training_checkpoint(
        path, expected={"type": "gru", "input_dim": 20, "num_classes": 3}
    )
    assert payload["checkpoint_kind"] == "training_state"
    assert payload["epoch"] == 2
    assert "optimizer_state" in payload
    assert training_meta["type"] == "gru"

    state, eval_meta = load_checkpoint(path, expected={"type": "gru", "input_dim": 20, "num_classes": 3})
    assert "rnn.weight_ih_l0" in state
    assert "optimizer_state" not in state
    assert eval_meta["pipeline"] == "sliding_window_temporal"


def test_checkpoint_keeps_training_dataset_provenance(tmp_path):
    cfg = {"type": "gru", "input_dim": 40, "num_classes": 6, "hidden": 8, "num_layers": 1}
    model = build_model(cfg)
    dataset = {
        "registered": True,
        "id": "temporal.actionmixed-v2",
        "version": "cleansight-actionmixed-v2",
        "revision": "b3cf7487",
        "feature_mapping": "actionmixed-bbox-8cls-v1",
        "labels": ["idle", "air_injection", "flush", "long_brush_insert", "long_brush_withdraw", "short_brush_cleaning"],
        "roles": {"train": "train"},
        "splits": {"train": {"fingerprint_sha256": "abc123"}},
    }
    meta = build_temporal_meta(
        cfg,
        {"dim": 40, "version": "actionmixed-bbox-8cls-v1"},
        pipeline="sliding_window_temporal",
        window=16,
        num_params=sum(p.numel() for p in model.parameters()),
        train_cfg={"epochs": 1},
        trained_at="t0",
        dataset=dataset,
    )
    path = tmp_path / "bound.pt"
    save_checkpoint(path, model.state_dict(), meta)

    _state, loaded = load_checkpoint(path)

    assert loaded["dataset"] == dataset
    assert_resume_dataset_compatible(loaded, dataset)
    changed = {**dataset, "splits": {"train": {"fingerprint_sha256": "changed"}}}
    with pytest.raises(ValueError, match="train split fingerprint"):
        assert_resume_dataset_compatible(loaded, changed)
