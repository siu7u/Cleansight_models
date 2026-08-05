from pathlib import Path

import pytest

from cleansight_eval.core.config import apply_overrides, load_config, validate_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_formal_config_rejects_missing_meta_switch():
    cfg = {
        "schema_version": 1,
        "pipeline": "full_sequence_temporal",
        "model": {"type": "mstcn", "allow_missing_meta": True},
        "data": {"name": "fixture", "root": "/tmp/fixture"},
        "evaluation": {"mode": "formal"},
    }

    with pytest.raises(ValueError, match="formal 评估不能启用"):
        validate_config(cfg)


def test_load_config_resolves_data_yaml_relative_to_config_file(tmp_path):
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    dataset = tmp_path / "datasets" / "group1" / "data.yaml"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("path: .\n", encoding="utf-8")
    cfg_path = cfg_dir / "exp.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "pipeline: detection",
                "model:",
                "  type: yolo",
                "data:",
                "  name: group1",
                "  data_yaml: ../datasets/group1/data.yaml",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["data"]["data_yaml"] == str(dataset.resolve())
    assert cfg["evaluation"]["mode"] == "formal"
    assert "evaluation.mode" in cfg["_config_provenance"]["default_fields"]


def test_load_config_keeps_absolute_data_root(tmp_path):
    root = tmp_path / "actionmixed"
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "pipeline: sliding_window_temporal",
                "model:",
                "  type: gru",
                "data:",
                "  name: actionmixed",
                f"  root: {root}",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["data"]["root"] == str(root)


def test_actionmixed_dataset_ref_resolves_canonical_root():
    cfg = load_config(REPO_ROOT / "framework" / "experiments" / "gru-actionmixed.yaml")

    assert cfg["data"]["dataset_ref"] == "temporal.actionmixed-v2"
    assert cfg["data"]["root"] == str((REPO_ROOT / "cleansight-ActionMixed").resolve())
    assert "data.root" not in cfg["_config_provenance"]["raw_fields"]


def test_dataset_ref_rejects_conflicting_explicit_root(tmp_path):
    cfg_path = tmp_path / "conflict.yaml"
    cfg_path.write_text(
        "schema_version: 1\npipeline: sliding_window_temporal\n"
        "model:\n  type: gru\n  input_dim: 40\n  num_classes: 6\n"
        "data:\n  dataset_ref: temporal.actionmixed-v2\n  root: wrong\n"
        "  split_train: train\n  split_val: val\n  split_eval: test\n"
        "feature_schema:\n  dim: 40\n  version: actionmixed-bbox-8cls-v1\n"
        "train:\n  epochs: 1\n  window: 16\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="与 dataset_ref=.*登记根目录"):
        load_config(cfg_path)


def test_unknown_field_and_override_are_rejected(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(
        "schema_version: 1\npipeline: detection\nmodel:\n  type: yolo\n  imsgz: 640\n"
        "data:\n  data_yaml: data.yaml\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="imsgz"):
        load_config(cfg_path)

    cfg_path.write_text(
        "schema_version: 1\npipeline: detection\nmodel:\n  type: yolo\n"
        "data:\n  data_yaml: data.yaml\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)
    overridden = apply_overrides(cfg, [("evaluation.conf", 0.25)])
    assert overridden["evaluation"]["conf"] == 0.25
    assert overridden["_config_provenance"]["override_fields"] == ["evaluation.conf"]
    with pytest.raises(ValueError, match="未知配置覆盖路径"):
        apply_overrides(cfg, [("evaluation.conff", 0.25)])


def test_feature_mask_targets_is_a_registered_config_parameter(tmp_path):
    cfg_path = tmp_path / "masked.yaml"
    cfg_path.write_text(
        "schema_version: 1\npipeline: sliding_window_temporal\n"
        "model:\n  type: gru\n"
        "data:\n  root: actionmixed\n"
        "feature_schema:\n  dim: 40\n  mask_targets: [syringe, air_gun]\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["feature_schema"]["mask_targets"] == ["syringe", "air_gun"]


def test_target_mask_augmentation_is_a_registered_config_section(tmp_path):
    cfg_path = tmp_path / "augmented.yaml"
    cfg_path.write_text(
        "schema_version: 1\npipeline: sliding_window_temporal\n"
        "model:\n  type: gru\n"
        "data:\n  root: actionmixed\n"
        "augmentation:\n  target_mask:\n    enabled: true\n"
        "    strategy: frame_dropout\n    targets: [syringe]\n    probability: 0.2\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg["augmentation"]["target_mask"]["targets"] == ["syringe"]
    assert cfg["augmentation"]["target_mask"]["probability"] == 0.2
