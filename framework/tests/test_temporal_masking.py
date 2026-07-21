"""ActionMixed 目标遮罩的特征层单元测试。"""

import numpy as np
import pytest

from cleansight_eval.temporal.data import (
    apply_target_mask_augmentation,
    featurize_frame_bbox,
    load_split,
    resolve_mask_target_ids,
    resolve_target_mask_augmentation,
    split_video_names,
)


def _write_detection_mapping(root):
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "data.yaml").write_text(
        "nc: 3\nnames:\n  0: hand\n  1: syringe\n  2: air_gun\n",
        encoding="utf-8",
    )


def test_featurize_frame_bbox_masks_only_selected_target(tmp_path):
    bbox_path = tmp_path / "frame.txt"
    bbox_path.write_text(
        "0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.2 0.3\n",
        encoding="utf-8",
    )

    feature = featurize_frame_bbox(
        bbox_path,
        n_classes=3,
        mask_target_ids=frozenset({1}),
    ).reshape(3, 5)

    np.testing.assert_allclose(feature[0], [1.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(feature[1], np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(feature[2], np.zeros(5, dtype=np.float32))


def test_resolve_mask_targets_accepts_names_and_ids(tmp_path):
    _write_detection_mapping(tmp_path)
    data_cfg = {"root": str(tmp_path), "frames_dir": "frames"}

    resolved = resolve_mask_target_ids(
        data_cfg,
        {"mask_targets": ["syringe", 2, "0"]},
    )

    assert resolved == frozenset({0, 1, 2})


def test_resolve_mask_targets_rejects_unknown_name(tmp_path):
    _write_detection_mapping(tmp_path)
    data_cfg = {"root": str(tmp_path), "frames_dir": "frames"}

    with pytest.raises(ValueError, match="未知遮罩目标"):
        resolve_mask_target_ids(data_cfg, {"mask_targets": ["unknown"]})


def test_no_mask_does_not_require_detection_mapping(tmp_path):
    data_cfg = {"root": str(tmp_path), "frames_dir": "frames"}

    assert resolve_mask_target_ids(data_cfg, {"dim": 40}) == frozenset()


def test_load_split_applies_configured_target_mask(tmp_path):
    _write_detection_mapping(tmp_path)
    (tmp_path / "labels" / "test").mkdir(parents=True)
    (tmp_path / "frames" / "test").mkdir(parents=True)
    (tmp_path / "labels" / "data.yaml").write_text(
        "nc: 1\nnames:\n  0: idle\n",
        encoding="utf-8",
    )
    (tmp_path / "labels" / "test" / "sample.mp4.txt").write_text(
        "1 0\n",
        encoding="utf-8",
    )
    (tmp_path / "frames" / "test" / "sample.mp4-000001.txt").write_text(
        "0 0.1 0.2 0.3 0.4\n1 0.5 0.6 0.2 0.3\n",
        encoding="utf-8",
    )
    data_cfg = {
        "root": str(tmp_path),
        "labels_dir": "labels",
        "frames_dir": "frames",
        "action_mapping": "labels/data.yaml",
    }

    features, truths, id2name = load_split(
        data_cfg,
        "test",
        feature_schema={"dim": 40, "mask_targets": ["syringe"]},
    )

    feature = features[0][0].reshape(8, 5)
    np.testing.assert_allclose(feature[0], [1.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(feature[1], np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(truths[0], [0])
    assert id2name == {0: "idle"}


def test_registered_dataset_loads_only_manifest_items(tmp_path, monkeypatch):
    """登记数据集不能因 split 目录新增文件而静默扩大训练样本。"""

    _write_detection_mapping(tmp_path)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "frames" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "data.yaml").write_text(
        "nc: 1\nnames:\n  0: idle\n", encoding="utf-8"
    )
    for name in ("registered.mp4", "unregistered.mp4"):
        (tmp_path / "labels" / "train" / f"{name}.txt").write_text(
            "1 0\n", encoding="utf-8"
        )
        (tmp_path / "frames" / "train" / f"{name}-000001.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

    import benchmark.core.testsets as testsets

    monkeypatch.setattr(testsets, "get_dataset_split", lambda _dataset, _split: object())
    monkeypatch.setattr(testsets, "read_split_items", lambda _spec: ["registered.mp4"])
    data_cfg = {
        "dataset_ref": "temporal.synthetic-v1",
        "root": str(tmp_path),
        "split_train": "train",
    }

    features, truths, _mapping = load_split(data_cfg, "train")

    assert len(features) == len(truths) == 1
    assert split_video_names(data_cfg, "train") == ["registered.mp4"]


def test_train_target_mask_probability_boundaries(tmp_path):
    _write_detection_mapping(tmp_path)
    data_cfg = {"root": str(tmp_path), "frames_dir": "frames"}
    features = [
        np.asarray(
            [
                [1.0, 0.1, 0.2, 0.3, 0.4, 1.0, 0.5, 0.6, 0.2, 0.3, 0, 0, 0, 0, 0],
                [1.0, 0.2, 0.3, 0.4, 0.5, 1.0, 0.6, 0.7, 0.3, 0.4, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    ]
    base = {
        "target_mask": {
            "enabled": True,
            "strategy": "frame_dropout",
            "targets": ["syringe"],
        }
    }

    clean = apply_target_mask_augmentation(
        features, data_cfg, {"target_mask": {**base["target_mask"], "probability": 0.0}}, seed=7
    )
    fully_masked = apply_target_mask_augmentation(
        features, data_cfg, {"target_mask": {**base["target_mask"], "probability": 1.0}}, seed=7
    )

    np.testing.assert_array_equal(clean[0], features[0])
    np.testing.assert_array_equal(fully_masked[0][:, :5], features[0][:, :5])
    np.testing.assert_array_equal(fully_masked[0][:, 5:10], np.zeros((2, 5), dtype=np.float32))


def test_train_target_mask_is_reproducible_with_seed(tmp_path):
    _write_detection_mapping(tmp_path)
    data_cfg = {"root": str(tmp_path), "frames_dir": "frames"}
    features = [np.ones((64, 15), dtype=np.float32)]
    augmentation = {
        "target_mask": {
            "enabled": True,
            "strategy": "frame_dropout",
            "targets": ["syringe"],
            "probability": 0.5,
        }
    }

    first = apply_target_mask_augmentation(features, data_cfg, augmentation, seed=42)
    second = apply_target_mask_augmentation(features, data_cfg, augmentation, seed=42)
    different = apply_target_mask_augmentation(features, data_cfg, augmentation, seed=43)

    np.testing.assert_array_equal(first[0], second[0])
    assert not np.array_equal(first[0], different[0])


def test_target_mask_augmentation_rejects_invalid_probability(tmp_path):
    _write_detection_mapping(tmp_path)
    data_cfg = {"root": str(tmp_path), "frames_dir": "frames"}

    with pytest.raises(ValueError, match="probability 必须在 0..1"):
        resolve_target_mask_augmentation(
            data_cfg,
            {
                "target_mask": {
                    "targets": ["syringe"],
                    "probability": 1.1,
                }
            },
        )
