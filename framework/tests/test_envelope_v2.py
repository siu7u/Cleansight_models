"""schema v2 的 testset、artifact、哈希和历史转换测试。"""

import json

from cleansight_eval.cli import upgrade_envelope
from cleansight_eval.core.artifacts import write_json_artifact
from cleansight_eval.core.provenance import build_checkpoint_info, build_run_info, resolve_testset_info
from cleansight_eval.temporal.artifacts import build_prediction_artifact


def test_temporal_artifact_is_hashed_and_recomputable(tmp_path):
    payload = build_prediction_artifact(
        {"video-a": ["idle", "flush"]},
        {"video-a": ["idle", "flush"]},
        ["idle", "flush"],
        window=None,
        inference_mode="full_sequence",
    )
    ref = write_json_artifact(tmp_path / "predictions.json", payload, relative_to=tmp_path)
    assert ref["path"] == "predictions.json"
    assert len(ref["sha256"]) == 64
    assert ref["recomputable"] is True


def test_checkpoint_info_contains_content_hash(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    info = build_checkpoint_info(checkpoint, tmp_path)
    assert info["path"] == "best.pt"
    assert len(info["sha256"]) == 64
    assert "size_bytes" not in info


def test_evaluation_run_info_excludes_environment_and_git(tmp_path):
    run_dir = tmp_path / "runs" / "gru-v1"
    checkpoint = run_dir / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "gru.yaml"
    config.write_text("pipeline: sliding_window_temporal\n", encoding="utf-8")

    info, found = build_run_info(checkpoint, config)

    assert found == run_dir
    assert set(info) == {"id", "created_at", "config"}
    assert "evaluation_environment" not in info
    assert "environment" not in info
    assert "git" not in info


def test_actionmixed_testset_is_registered_and_leak_is_explicit():
    cfg = {
        "pipeline": "full_sequence_temporal",
        "data": {"name": "cleansight-ActionMixed", "split_eval": "test"},
        "feature_schema": {"dim": 40},
        "evaluation": {"testset_id": "temporal.actionmixed-v1.test"},
    }
    info = resolve_testset_info(cfg)
    assert info["registered"] is True
    assert info["split"] == "test"
    assert len(info["fingerprint_sha256"]) == 64
    assert any("train/test 泄漏" in error for error in info["validation_errors"])


def test_upgrade_v1_writes_new_v2_file_without_overwrite(tmp_path):
    source = tmp_path / "old.envelope.json"
    source.write_text(
        json.dumps(
            {
                "model_type": "gru",
                "model_id": "gru-old",
                "pipeline": "sliding_window_temporal",
                "checkpoint": "missing.pt",
                "dataset": "legacy-data",
                "metrics": {
                    "acc": {
                        "state": "computed",
                        "value": 50.0,
                        "spec": "acc/v1",
                        "reason": None,
                    }
                },
                "performance": {},
            }
        ),
        encoding="utf-8",
    )
    output = upgrade_envelope.main(["--input", str(source)])
    converted = json.loads(open(output, encoding="utf-8").read())
    assert source.exists()
    assert converted["schema_version"] == 2
    assert converted["model"]["id"] == "gru-old"
    assert converted["integrity"]["ok"] is False
