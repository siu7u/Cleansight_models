"""tools/visualize_predictions.py：时序预测产物 → 动作阶段预览视频测试。

验收标准（verification-first）：
- 顶部预测横幅按动作色填充（像素级验证），下方 GT 深色小字条
- 预测序列与数据集标签帧严格对齐（长度不一致报错，防静默错渲）
- 数据集缺失的序列跳过告警；--sequence / --max-frames / --no-boxes 生效
- 像素源二选一（--images / --video），与 visualize_dataset 同语义
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.visualize_predictions import (  # noqa: E402
    load_artifact,
    main,
    parse_args,
    render_artifact_videos,
    render_prediction_video,
)
from tools.visualize_dataset import CLASS_COLORS  # noqa: E402

GRAY = 128


def _make_artifact(path: Path, sequences: dict) -> dict:
    """构造预测产物：``sequences`` 为 ``{序列名: ([pred_id...], [truth_id...])}``。"""

    labels = [
        {"id": 0, "name": "idle"},
        {"id": 1, "name": "air_injection"},
        {"id": 2, "name": "flush"},
    ]
    items = {}
    for name, (pred_ids, truth_ids) in sequences.items():
        items[name] = {
            "prediction_start_frame": 0,
            "num_predictions": len(pred_ids),
            "predicted_label_ids": pred_ids,
            "truth_label_ids": truth_ids,
            "predicted_labels": ["x"] * len(pred_ids),
            "truth_labels": ["x"] * len(truth_ids),
        }
    artifact = {
        "schema_version": 1,
        "task_type": "temporal",
        "prediction_format": "frame_labels",
        "inference": {"mode": "full_sequence"},
        "labels": labels,
        "items": items,
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact


def _make_dataset(root: Path, seq: str = "demo.mp4") -> None:
    """构造持久化训练数据（labels/<split>/ + frames/<split>/ + 帧图），3 个标签帧。"""

    labels = root / "labels" / "train"
    frames = root / "frames" / "train"
    images = root / "images"
    for frame in (1, 5, 9):
        (images / f"{seq}-{frame:06d}.jpg").parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(images / f"{seq}-{frame:06d}.jpg"), np.full((128, 128, 3), GRAY, dtype=np.uint8))
    (labels / f"{seq}.txt").parent.mkdir(parents=True, exist_ok=True)
    (labels / f"{seq}.txt").write_text("1 0\n5 1\n9 2\n", encoding="utf-8")
    (frames / f"{seq}-000001.txt").parent.mkdir(parents=True, exist_ok=True)
    (frames / f"{seq}-000001.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (frames / f"{seq}-000005.txt").write_text("", encoding="utf-8")


def _make_video(path: Path, frames: int = 10) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 128))
    for _ in range(frames):
        writer.write(np.full((128, 128, 3), GRAY, dtype=np.uint8))
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


def _banner_color(frame: np.ndarray) -> np.ndarray:
    """顶部预测横幅的取样颜色（避开文字，取横幅中部左侧）。"""

    return frame[2, 60]


def _gt_strip_gray(frame: np.ndarray) -> bool:
    """GT 小字条（y 30..50）为深色底：三通道都接近 60。"""

    b, g, r = (int(v) for v in frame[40, 40])
    return 20 <= b <= 110 and 20 <= g <= 110 and 20 <= r <= 110


class TestLoadArtifact:
    def test_missing_fields_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        with pytest.raises(ValueError, match="labels/items"):
            load_artifact(path)


class TestRenderPredictionVideo:
    def test_images_source_banner_and_gt_strip(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        artifact = _make_artifact(tmp_path / "pred.json", {"demo.mp4": ([1, 2, 0], [0, 2, 0])})
        output = render_prediction_video(
            artifact, root, "demo.mp4",
            images_dir=root / "images",
            out_path=tmp_path / "pred.mp4",
        )
        assert output.is_file()
        frames = _read_video_frames(output)
        assert len(frames) == 3
        # 帧 1 预测动作 1 → 横幅 CLASS_COLORS[1]（蓝）；GT 小字条深色底
        banner = _banner_color(frames[0])
        blue = CLASS_COLORS[1]
        assert int(banner[0]) - int(banner[1]) > 60  # B 明显大于 R
        assert _gt_strip_gray(frames[0])
        # 帧 3 预测动作 0 → 横幅 CLASS_COLORS[0]（绿）
        green = _banner_color(frames[2])
        assert int(green[1]) - int(green[2]) > 60  # G 明显大于 R

    def test_mismatch_frame_renders(self, tmp_path, capsys):
        root = tmp_path / "ds"
        _make_dataset(root)
        artifact = _make_artifact(tmp_path / "pred.json", {"demo.mp4": ([1, 1, 0], [0, 2, 0])})
        render_prediction_video(
            artifact, root, "demo.mp4",
            images_dir=root / "images",
            out_path=tmp_path / "pred.mp4",
        )
        log = capsys.readouterr().out
        # 帧 1/2 预测 1 但真值 0/2 → frame-acc = 1/3
        assert "frame-acc=33.3%" in log

    def test_length_mismatch_errors(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        artifact = _make_artifact(tmp_path / "pred.json", {"demo.mp4": ([0, 1], [0, 1])})
        with pytest.raises(ValueError, match="长度"):
            render_prediction_video(
                artifact, root, "demo.mp4",
                images_dir=root / "images",
                out_path=tmp_path / "pred.mp4",
            )

    def test_unknown_sequence_errors(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        artifact = _make_artifact(tmp_path / "pred.json", {"demo.mp4": ([0, 0, 0], [0, 0, 0])})
        with pytest.raises(KeyError, match="没有序列"):
            render_prediction_video(
                artifact, root, "nope.mp4",
                images_dir=root / "images",
                out_path=tmp_path / "pred.mp4",
            )

    def test_no_pixel_source_errors(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        artifact = _make_artifact(tmp_path / "pred.json", {"demo.mp4": ([0, 0, 0], [0, 0, 0])})
        with pytest.raises(ValueError, match="--images 或 --video"):
            render_prediction_video(
                artifact, root, "demo.mp4", out_path=tmp_path / "pred.mp4"
            )

    def test_video_source_extracts_by_frame_number(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        artifact = _make_artifact(tmp_path / "pred.json", {"demo.mp4": ([1, 2, 0], [0, 2, 0])})
        video = _make_video(tmp_path / "demo.mp4", frames=10)
        render_prediction_video(
            artifact, root, "demo.mp4",
            video_path=video,
            out_path=tmp_path / "pred_video.mp4",
        )
        assert len(_read_video_frames(tmp_path / "pred_video.mp4")) == 3

    def test_max_frames_truncates(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        artifact = _make_artifact(tmp_path / "pred.json", {"demo.mp4": ([1, 2, 0], [0, 2, 0])})
        render_prediction_video(
            artifact, root, "demo.mp4",
            images_dir=root / "images",
            out_path=tmp_path / "pred.mp4",
            max_frames=2,
        )
        assert len(_read_video_frames(tmp_path / "pred.mp4")) == 2

    def test_no_boxes_flag(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        artifact = _make_artifact(tmp_path / "pred.json", {"demo.mp4": ([0, 0, 0], [0, 0, 0])})
        render_prediction_video(
            artifact, root, "demo.mp4",
            images_dir=root / "images",
            out_path=tmp_path / "pred_nobox.mp4",
            draw_boxes=False,
        )
        frame = _read_video_frames(tmp_path / "pred_nobox.mp4")[0]
        # 帧 1 的框左边界（x=17, y=40）应保持原图灰（未画框）
        assert abs(int(frame[60, 33, 0]) - GRAY) < 40


class TestRenderArtifactVideos:
    def test_missing_dataset_item_skipped(self, tmp_path, capsys):
        root = tmp_path / "ds"
        _make_dataset(root)
        artifact = _make_artifact(
            tmp_path / "pred.json",
            {"demo.mp4": ([1, 2, 0], [0, 2, 0]), "missing.mp4": ([0, 0, 0], [0, 0, 0])},
        )
        outputs = render_artifact_videos(
            artifact, root,
            images_dir=root / "images",
            out_dir=tmp_path / "out",
        )
        assert len(outputs) == 1
        assert (tmp_path / "out" / "demo_pred.mp4").is_file()
        assert "跳过（数据集 labels/ 中无此序列）: missing.mp4" in capsys.readouterr().out

    def test_sequence_filter(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root, seq="a.mp4")
        _make_dataset(root, seq="b.mp4")
        artifact = _make_artifact(
            tmp_path / "pred.json",
            {"a.mp4": ([0, 0, 0], [0, 0, 0]), "b.mp4": ([0, 0, 0], [0, 0, 0])},
        )
        outputs = render_artifact_videos(
            artifact, root,
            images_dir=root / "images",
            out_dir=tmp_path / "out",
            sequence="b.mp4",
        )
        assert len(outputs) == 1 and outputs[0].name == "b_pred.mp4"


class TestCli:
    def test_parse_args(self):
        args = parse_args(
            ["--artifact", "a.json", "--dataset", "ds", "--images", "img",
             "--out-dir", "out", "--sequence", "b.mp4", "--no-boxes"]
        )
        assert args.sequence == "b.mp4" and args.no_boxes is True

    def test_main_renders_all(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root)
        artifact_path = tmp_path / "pred.json"
        _make_artifact(artifact_path, {"demo.mp4": ([1, 2, 0], [0, 2, 0])})
        code = main(
            ["--artifact", str(artifact_path), "--dataset", str(root),
             "--images", str(root / "images"), "--out-dir", str(tmp_path / "out")]
        )
        assert code == 0
        assert (tmp_path / "out" / "demo_pred.mp4").is_file()
