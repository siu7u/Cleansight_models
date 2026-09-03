"""整帧图像 embedding 提取工具的纯函数单元测试（不加载模型权重）。"""

import cv2
import numpy as np

from cleansight_eval.temporal.features.extract_embeddings import (
    _frame_image_path,
    load_frame_rgb,
)


def _write_jpg(path, size=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(size)[None, :]  # 渐变图，内容可区分
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    buf.tofile(str(path))


def test_frame_image_path_aligns_with_label_frame_id(tmp_path):
    images_dir = tmp_path / "images"
    stem = "clip_123.mp4"
    path = _frame_image_path(images_dir, stem, 5)

    assert path == images_dir / "clip_123.mp4-000005.jpg"


def test_load_frame_rgb_shape_and_channel_order(tmp_path):
    jpg = tmp_path / "frame.jpg"
    _write_jpg(jpg)

    frame = load_frame_rgb(jpg, size=32)

    assert frame.shape == (32, 32, 3)
    assert frame.dtype == np.uint8
    # BGR->RGB：写图时 B 通道是渐变、R 通道为 0；读回后 RGB 第 0 通道应接近 0
    assert float(frame[..., 0].mean()) < 1.0
    assert float(frame[..., 2].mean()) > 30.0


def test_load_frame_rgb_missing_file_returns_none(tmp_path):
    assert load_frame_rgb(tmp_path / "missing.jpg", 32) is None
