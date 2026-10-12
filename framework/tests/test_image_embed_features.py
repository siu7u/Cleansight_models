"""图像 embedding 契约（actionmixed-bbox-embed-*，形态 B）的单元测试。"""

import numpy as np
import pytest

from cleansight_eval.temporal.data import (
    apply_target_mask_augmentation,
    load_split,
    resolve_image_feature_dim,
)
from cleansight_eval.temporal.features import (
    EMBED_BBOX_DIM,
    IMAGE_EMBED_VERSION,
    load_video_embeddings,
    resolve_embed_dim,
)
from cleansight_eval.temporal.models import build_model

TEST_EMBED_DIM = 4  # 单元测试用小维度（真实契约 576）
TEST_FEATURE_DIM = EMBED_BBOX_DIM + TEST_EMBED_DIM  # = 44


def _write_dataset(tmp_path, embed_rows=2, embed_dim=TEST_EMBED_DIM):
    """搭一份最小数据集：labels/ + frames/ + embeddings/（与 extract_embeddings 产物同构）。"""

    (tmp_path / "labels" / "test").mkdir(parents=True)
    (tmp_path / "frames" / "test").mkdir(parents=True)
    (tmp_path / "embeddings" / "test").mkdir(parents=True)
    (tmp_path / "labels" / "data.yaml").write_text("nc: 1\nnames:\n  0: idle\n", encoding="utf-8")
    (tmp_path / "labels" / "test" / "sample.mp4.txt").write_text("1 0\n2 0\n", encoding="utf-8")
    (tmp_path / "frames" / "test" / "sample.mp4-000001.txt").write_text(
        "0 0.1 0.2 0.3 0.4\n", encoding="utf-8"
    )
    (tmp_path / "frames" / "test" / "sample.mp4-000002.txt").write_text("", encoding="utf-8")
    embeddings = np.arange(embed_rows * embed_dim, dtype=np.float32).reshape(embed_rows, embed_dim)
    np.save(tmp_path / "embeddings" / "test" / "sample.mp4.npy", embeddings)
    return embeddings


def _data_cfg(tmp_path):
    return {
        "root": str(tmp_path),
        "labels_dir": "labels",
        "frames_dir": "frames",
        "action_mapping": "labels/data.yaml",
        "embedding_root": str(tmp_path / "embeddings"),
    }


def test_load_split_embed_contract_appends_image_block(tmp_path):
    """load_split 输出 [T, 40+embed]：左 bbox 块逐帧编码，右图像块按位置对齐 npy。"""

    embeddings = _write_dataset(tmp_path)

    features, truths, id2name = load_split(
        _data_cfg(tmp_path),
        "test",
        feature_schema={"dim": TEST_FEATURE_DIM, "version": IMAGE_EMBED_VERSION},
    )

    assert features[0].shape == (2, TEST_FEATURE_DIM)
    # 第 1 帧：bbox 块为唯一框的 [presence, cx, cy, w, h]，图像块为 npy 第 0 行
    np.testing.assert_allclose(features[0][0, :5], [1.0, 0.1, 0.2, 0.3, 0.4], rtol=1e-6)
    np.testing.assert_array_equal(features[0][0, EMBED_BBOX_DIM:], embeddings[0])
    # 第 2 帧：空 bbox 文件 → bbox 块全零，图像块仍来自 npy（图像通道不因无检测而清零）
    np.testing.assert_array_equal(features[0][1, :EMBED_BBOX_DIM], np.zeros(EMBED_BBOX_DIM))
    np.testing.assert_array_equal(features[0][1, EMBED_BBOX_DIM:], embeddings[1])
    np.testing.assert_array_equal(truths[0], [0, 0])
    assert id2name == {0: "idle"}


def test_load_split_embed_smoke_truncation_keeps_prefix(tmp_path):
    """max_frames（smoke 截断）时只取标签前缀对应的 embedding 行，不报行数不符。"""

    _write_dataset(tmp_path)

    features, _truths, _id2name = load_split(
        _data_cfg(tmp_path),
        "test",
        feature_schema={"dim": TEST_FEATURE_DIM, "version": IMAGE_EMBED_VERSION},
        max_frames=1,
    )

    assert features[0].shape == (1, TEST_FEATURE_DIM)


def test_load_split_embed_rejects_row_mismatch(tmp_path):
    """embedding 行数与标签行数不一致（产物陈旧）立即报错，不静默错位。"""

    _write_dataset(tmp_path, embed_rows=3)

    with pytest.raises(ValueError, match="行数"):
        load_split(
            _data_cfg(tmp_path),
            "test",
            feature_schema={"dim": TEST_FEATURE_DIM, "version": IMAGE_EMBED_VERSION},
        )


def test_load_split_embed_rejects_dim_mismatch(tmp_path):
    """embedding 列数与契约推导的图像块宽度不一致立即报错。"""

    _write_dataset(tmp_path, embed_dim=TEST_EMBED_DIM + 1)

    with pytest.raises(ValueError, match="维度"):
        load_split(
            _data_cfg(tmp_path),
            "test",
            feature_schema={"dim": TEST_FEATURE_DIM, "version": IMAGE_EMBED_VERSION},
        )


def test_load_split_embed_requires_embedding_root(tmp_path):
    """缺 data.embedding_root 时直接报错（提示登记 feature_embed.root）。"""

    _write_dataset(tmp_path)
    data_cfg = _data_cfg(tmp_path)
    data_cfg.pop("embedding_root")

    with pytest.raises(ValueError, match="embedding_root"):
        load_split(
            data_cfg,
            "test",
            feature_schema={"dim": TEST_FEATURE_DIM, "version": IMAGE_EMBED_VERSION},
        )


def test_load_video_embeddings_missing_file(tmp_path):
    """产物文件缺失时报出待生成的路径。"""

    with pytest.raises(FileNotFoundError, match="extract_embeddings"):
        load_video_embeddings(tmp_path, "test", "absent.mp4", TEST_EMBED_DIM, expected_rows=1)


def test_resolve_embed_dim_rejects_bbox_only_dim():
    """契约维度必须大于 bbox 块宽度，否则不是图像契约。"""

    with pytest.raises(ValueError, match="feature_schema.dim"):
        resolve_embed_dim({"dim": EMBED_BBOX_DIM})


def test_augmentation_masks_only_bbox_block_for_embed_contract(tmp_path):
    """图像契约下目标遮罩只清零 bbox 的 5 维类块，图像块保持原样。"""

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)
    names = ("hand", "scope_control_body", "scope_mid_section", "scope_distal_end",
             "syringe", "air_gun", "short_brush", "brush_tip_out")
    lines = [f"nc: {len(names)}", "names:"]
    for i, name in enumerate(names):
        lines.append(f"  {i}: {name}")
    (frames_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    features = [np.ones((2, TEST_FEATURE_DIM), dtype=np.float32)]
    augmentation = {
        "target_mask": {
            "enabled": True,
            "strategy": "frame_dropout",
            "targets": ["syringe"],  # 检测类 ID=4 → bbox 块 [20:25]
            "probability": 1.0,
        }
    }

    masked = apply_target_mask_augmentation(
        features,
        {"root": str(tmp_path), "frames_dir": "frames"},
        augmentation,
        seed=7,
        feature_schema={"dim": TEST_FEATURE_DIM, "version": IMAGE_EMBED_VERSION},
    )

    np.testing.assert_array_equal(masked[0][:, 20:25], np.zeros((2, 5), dtype=np.float32))
    assert masked[0][:, :20].sum() == 2 * 20
    assert masked[0][:, 25:EMBED_BBOX_DIM].sum() == 2 * 15
    assert masked[0][:, EMBED_BBOX_DIM:].sum() == 2 * TEST_EMBED_DIM  # 图像块不被遮罩


def test_resolve_image_feature_dim_cross_checks_model(tmp_path):
    """图像契约与 model.image_dim 双向交叉校验：一致放行，不一致/越界声明报错。"""

    embed_schema = {"dim": TEST_FEATURE_DIM, "version": IMAGE_EMBED_VERSION}

    assert resolve_image_feature_dim({"image_dim": TEST_EMBED_DIM}, embed_schema) == TEST_EMBED_DIM
    assert resolve_image_feature_dim({}, {"dim": 40, "version": "actionmixed-bbox-8cls-v1"}) == 0

    with pytest.raises(ValueError, match="image_dim"):
        resolve_image_feature_dim({"image_dim": TEST_EMBED_DIM + 1}, embed_schema)
    with pytest.raises(ValueError, match="只在图像 embedding 契约下有效"):
        resolve_image_feature_dim({"image_dim": TEST_EMBED_DIM}, {"dim": 40, "version": "actionmixed-bbox-8cls-v1"})


def test_gru_image_projection_shapes():
    """GRU 在 image_dim>0 时先投影图像块再拼接，输出形状不变。"""

    import torch

    model = build_model({
        "type": "gru",
        "input_dim": TEST_FEATURE_DIM,
        "num_classes": 3,
        "hidden": 8,
        "num_layers": 1,
        "image_dim": TEST_EMBED_DIM,
        "image_proj_dim": 2,
    })
    assert model.rnn.input_size == EMBED_BBOX_DIM + 2  # 40 bbox + 2 维投影

    out = model(torch.zeros(2, 5, TEST_FEATURE_DIM))

    assert out.shape == (2, 5, 3)


def test_build_model_rejects_image_dim_for_other_models():
    """非 GRU 架构声明 image_dim 直接报错，避免投影头被静默忽略。"""

    with pytest.raises(ValueError, match="只在 GRU 上实现"):
        build_model({
            "type": "mstcn",
            "input_dim": TEST_FEATURE_DIM,
            "num_classes": 3,
            "image_dim": TEST_EMBED_DIM,
        })


def test_gru_rejects_invalid_image_dim():
    """image_dim 必须落在 (0, input_dim) 内。"""

    with pytest.raises(ValueError, match="image_dim"):
        build_model({
            "type": "gru",
            "input_dim": TEST_FEATURE_DIM,
            "num_classes": 3,
            "image_dim": TEST_FEATURE_DIM,
        })
