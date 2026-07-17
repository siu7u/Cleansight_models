from pathlib import Path

import pytest

from cleansight_eval.core.config import apply_overrides, load_config


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
