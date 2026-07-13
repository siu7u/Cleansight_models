from pathlib import Path

from cleansight_eval.core.config import load_config


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


def test_load_config_keeps_absolute_data_root(tmp_path):
    root = tmp_path / "actionmixed"
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(
        "\n".join(
            [
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
