"""tools/visualize_dataset.py：持久化训练数据可视化测试。

验收标准（verification-first）：
- 图片帧源：labels/frames 持久化产物 → 预览视频（帧数 = 标签帧数，框画在帧上）
- 视频源：按真实帧号抽取帧图，同样产出预览视频
- 缺 bbox 文件告警不中断；无像素源 / 未知序列 / 空标签明确报错
- CLI：--sequence 缺省取 split 第一个序列；--max-frames 截断
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.visualize_dataset import (  # noqa: E402
    find_frame_image,
    load_class_names,
    load_frame_boxes,
    load_label_frames,
    main,
    parse_args,
    render_dataset_preview,
)

GRAY = 128


def _make_dataset(root: Path, seq: str = "demo.mp4") -> None:
    """构造一个 split 的持久化训练数据（labels/<split>/ + frames/<split>/ + 帧图）。

    root 即数据集根（与 datasets/cleansight-ActionMixed-auto 同构布局），
    images 为外部像素源目录（对应 datasets/.../images/<split>/）。
    """

    labels = root / "labels" / "train"
    frames = root / "frames" / "train"
    images = root / "images"
    for frame in (1, 5, 9):
        (images / f"{seq}-{frame:06d}.jpg").parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(images / f"{seq}-{frame:06d}.jpg"), np.full((64, 64, 3), GRAY, dtype=np.uint8))
    (labels / f"{seq}.txt").parent.mkdir(parents=True, exist_ok=True)
    (labels / f"{seq}.txt").write_text("1 0\n5 2\n9 4\n", encoding="utf-8")
    (frames / f"{seq}-000001.txt").parent.mkdir(parents=True, exist_ok=True)
    # 帧 1：hand(0) 大框 + syringe(4)；帧 5：hand 小框；帧 9：无 bbox 文件（缺产物告警路径）
    (frames / f"{seq}-000001.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n4 0.1 0.1 0.1 0.1\n", encoding="utf-8"
    )
    (frames / f"{seq}-000005.txt").write_text("0 0.3 0.3 0.2 0.2\n", encoding="utf-8")
    (root / "labels" / "data.yaml").write_text(
        yaml.safe_dump({"nc": 6, "names": {0: "idle", 2: "brush", 4: "flush"}}, allow_unicode=True),
        encoding="utf-8",
    )
    (root / "frames" / "data.yaml").write_text(
        yaml.safe_dump({"nc": 8, "names": {0: "hand", 4: "syringe"}}, allow_unicode=True),
        encoding="utf-8",
    )


def _make_video(path: Path, frames: int = 10) -> Path:
    """合成 10fps 灰度视频（帧号 1-based 对应 label 帧号）。"""

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for _ in range(frames):
        writer.write(np.full((64, 64, 3), GRAY, dtype=np.uint8))
    writer.release()
    return path


def _read_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _box_border_pixel_green(frame: np.ndarray) -> bool:
    """帧 1 大框（cx=0.5, cy=0.5, w=0.5, h=0.5 → 像素 16..48）左边界上是否有绿色。"""

    g = int(frame[40, 17, 1])
    r = int(frame[40, 17, 2])
    return g - r > 60  # 绿色框 BGR=(0,255,0)；阈值留 mp4 有损解码余量


class TestParsing:
    def test_load_label_frames_skips_invalid_lines(self, tmp_path):
        path = tmp_path / "labels.txt"
        path.write_text("1 0\nbad line\n9 4\n", encoding="utf-8")
        assert load_label_frames(path) == [(1, 0), (9, 4)]

    def test_load_frame_boxes_missing_file_returns_empty(self, tmp_path):
        assert load_frame_boxes(tmp_path / "nope.txt") == []

    def test_load_class_names_fallback_and_yaml(self, tmp_path):
        missing = tmp_path / "nope.yaml"
        assert load_class_names(missing, ["a", "b"]) == {0: "a", 1: "b"}
        path = tmp_path / "names.yaml"
        path.write_text("names:\n  2: brush\n", encoding="utf-8")
        assert load_class_names(path, ["a"]) == {2: "brush"}

    def test_find_frame_image(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        (images / "demo.mp4-000001.jpg").write_bytes(b"x")
        assert find_frame_image(images, "demo.mp4", 1) == images / "demo.mp4-000001.jpg"
        assert find_frame_image(images, "demo.mp4", 2) is None


class TestRenderDatasetPreview:
    def test_images_source_draws_boxes(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        output = render_dataset_preview(
            root,
            "train",
            "demo.mp4",
            images_dir=root / "images",
            output_path=tmp_path / "preview.mp4",
        )
        assert output.is_file()
        frames = _read_video_frames(output)
        assert len(frames) == 3  # 只渲染标签帧 1/5/9
        assert frames[0].shape[:2] == (64, 64)
        # 帧 1 有 hand 大框（左边界绿色）；帧 9 无 bbox 文件 → 无检测框（无绿色）
        assert _box_border_pixel_green(frames[0])
        assert not _box_border_pixel_green(frames[2])

    def test_video_source_extracts_by_frame_number(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        video = _make_video(tmp_path / "demo.mp4", frames=10)
        output = render_dataset_preview(
            root,
            "train",
            "demo.mp4",
            video_path=video,
            output_path=tmp_path / "preview_video.mp4",
        )
        frames = _read_video_frames(output)
        assert len(frames) == 3
        assert _box_border_pixel_green(frames[0])
        assert not _box_border_pixel_green(frames[2])  # 帧 9 无 bbox

    def test_missing_box_file_warns_but_continues(self, tmp_path, capsys):
        root = tmp_path / "ds"
        _make_dataset(root)
        render_dataset_preview(
            root,
            "train",
            "demo.mp4",
            images_dir=root / "images",
            output_path=tmp_path / "preview.mp4",
        )
        log = capsys.readouterr().out
        assert "无 bbox 文件" in log and "缺 bbox 文件帧数: 1" in log

    def test_missing_pixel_frame_skipped(self, tmp_path, capsys):
        root = tmp_path / "ds"
        _make_dataset(root)
        # 删掉帧 9 的图片 → 该帧跳过，输出只剩 2 帧
        (root / "images" / "demo.mp4-000009.jpg").unlink()
        render_dataset_preview(
            root,
            "train",
            "demo.mp4",
            images_dir=root / "images",
            output_path=tmp_path / "preview.mp4",
        )
        assert "跳过（缺像素帧 9）" in capsys.readouterr().out
        assert len(_read_video_frames(tmp_path / "preview.mp4")) == 2

    def test_max_frames_truncates(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        render_dataset_preview(
            root,
            "train",
            "demo.mp4",
            images_dir=root / "images",
            output_path=tmp_path / "preview.mp4",
            max_frames=2,
        )
        assert len(_read_video_frames(tmp_path / "preview.mp4")) == 2

    def test_no_pixel_source_errors(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        with pytest.raises(ValueError, match="--images 或 --video"):
            render_dataset_preview(
                root, "train", "demo.mp4", output_path=tmp_path / "x.mp4"
            )

    def test_unknown_sequence_errors_with_available(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root, seq="demo.mp4")
        _make_dataset(root, seq="other.mp4")
        with pytest.raises(FileNotFoundError, match="可用"):
            render_dataset_preview(
                root,
                "train",
                "missing.mp4",
                images_dir=root / "images",
                output_path=tmp_path / "x.mp4",
            )

    def test_empty_labels_errors(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        (root / "labels" / "train" / "demo.mp4.txt").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="动作标签为空"):
            render_dataset_preview(
                root,
                "train",
                "demo.mp4",
                images_dir=root / "images",
                output_path=tmp_path / "x.mp4",
            )


class TestCli:
    def test_parse_args(self):
        args = parse_args(
            ["--dataset", "ds", "--split", "val", "--sequence", "a.mp4",
             "--images", "img", "--output", "out.mp4", "--max-frames", "5"]
        )
        assert args.split == "val" and args.sequence == "a.mp4"
        assert args.max_frames == 5

    def test_main_sequence_defaults_to_first(self, tmp_path, capsys):
        root = tmp_path / "ds"
        _make_dataset(root, seq="b.mp4")
        _make_dataset(root, seq="a.mp4")
        code = main(
            ["--dataset", str(root), "--split", "train",
             "--images", str(root / "images"),
             "--output", str(tmp_path / "preview.mp4")]
        )
        assert code == 0
        assert "a.mp4" in capsys.readouterr().out  # 取第一个排序序列
        assert len(_read_video_frames(tmp_path / "preview.mp4")) == 3
