"""tools/predict_timeline.py：.pt → 动作段时间线 + 带状图 + 预测视频 测试。

验收标准（verification-first）：
- derive_segments：相邻同类抽样帧合并成段，起止帧号/帧数正确，未知类别兜底命名
- build_prediction_artifact：PredictionOutput → prediction-artifact-v1 结构（ids 回编）
- _resolve_split：显式 split / 序列自动探测 / 缺省 train / 未命中报错
- 全链路 smoke（真实权重 + 真实数据集）：时间线 JSON + 带状图 + 视频三样产物齐全
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.predict_timeline import (  # noqa: E402
    build_cfg,
    build_prediction_artifact,
    derive_segments,
    load_meta,
    parse_args,
    run_predict_timeline,
    _resolve_split,
)


class TestLoadMeta:
    def test_missing_sidecar_errors(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"x")
        with pytest.raises(FileNotFoundError, match="meta sidecar"):
            load_meta(ckpt)

    def test_reads_sidecar(self, tmp_path):
        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"x")
        (tmp_path / "best.pt.meta.json").write_text(
            '{"type": "mstcn", "pipeline": "full_sequence_temporal"}', encoding="utf-8"
        )
        meta = load_meta(ckpt)
        assert meta["type"] == "mstcn"


class TestBuildCfg:
    def test_cfg_from_meta(self):
        meta = {
            "model": {"type": "mstcn", "input_dim": 40, "num_classes": 6},
            "feature_schema": {"dim": 40, "version": "actionmixed-bbox-8cls-v1"},
            "window": None,
        }
        cfg = build_cfg(meta, Path("/ds"), "test")
        assert cfg["model"] == meta["model"]
        assert cfg["data"]["root"] == "/ds"
        assert cfg["data"]["split_eval"] == "test"
        assert cfg["evaluation"]["mode"] == "formal"
        assert cfg["train"]["window"] == 64  # meta 无 window 时兜底


class TestDeriveSegments:
    def test_merges_adjacent_same_action(self):
        id2name = {0: "idle", 1: "air_injection", 2: "flush"}
        segments = derive_segments([0, 0, 1, 1, 1, 0, 2], [1, 5, 9, 13, 17, 21, 25], id2name)
        assert segments == [
            {"action": "idle", "action_id": 0, "start_frame": 1, "end_frame": 5, "num_frames": 2},
            {"action": "air_injection", "action_id": 1, "start_frame": 9, "end_frame": 17, "num_frames": 3},
            {"action": "idle", "action_id": 0, "start_frame": 21, "end_frame": 21, "num_frames": 1},
            {"action": "flush", "action_id": 2, "start_frame": 25, "end_frame": 25, "num_frames": 1},
        ]

    def test_unknown_id_falls_back(self):
        segments = derive_segments([9], [1], {0: "idle"})
        assert segments[0]["action"] == "cls_9"


class TestBuildPredictionArtifact:
    def _output(self):
        return SimpleNamespace(
            predictions={
                "a.mp4": ["idle", "flush", "idle"],
                "b.mp4": ["air_injection"],
            },
            targets={"a.mp4": ["idle", "flush", "flush"]},
            labels=["idle", "air_injection", "flush"],
            pipeline="full_sequence_temporal",
        )

    def test_artifact_schema(self):
        artifact = build_prediction_artifact(self._output(), ["a.mp4"])
        assert artifact["schema_version"] == 1
        assert artifact["task_type"] == "temporal"
        assert artifact["labels"] == [
            {"id": 0, "name": "idle"},
            {"id": 1, "name": "air_injection"},
            {"id": 2, "name": "flush"},
        ]
        item = artifact["items"]["a.mp4"]
        assert item["predicted_label_ids"] == [0, 2, 0]
        assert item["truth_label_ids"] == [0, 2, 2]
        assert item["num_predictions"] == 3


class TestResolveSplit:
    def _dataset(self, root: Path) -> Path:
        for split in ("train", "test"):
            (root / "labels" / split).mkdir(parents=True)
        (root / "labels" / "test" / "a.mp4.txt").write_text("1 0\n", encoding="utf-8")
        return root

    def test_explicit_split_wins(self, tmp_path):
        root = self._dataset(tmp_path)
        assert _resolve_split(root, "a.mp4", "test") == "test"

    def test_sequence_auto_detect_unique(self, tmp_path):
        root = self._dataset(tmp_path)
        assert _resolve_split(root, "a.mp4", None) == "test"

    def test_no_sequence_defaults_train(self, tmp_path):
        root = self._dataset(tmp_path)
        assert _resolve_split(root, None, None) == "train"

    def test_unknown_sequence_errors(self, tmp_path):
        root = self._dataset(tmp_path)
        with pytest.raises(FileNotFoundError, match="找不到序列"):
            _resolve_split(root, "nope.mp4", None)


class TestCli:
    def test_parse_args(self):
        args = parse_args(
            ["--ckpt", "best.pt", "--dataset", "ds", "--sequence", "a.mp4",
             "--out-dir", "out", "--no-boxes"]
        )
        assert args.sequence == "a.mp4" and args.no_boxes is True


class TestIntegrationSmoke:
    """真实权重 + 真实数据集的全链路（ultralytics 无关，仅 torch）。"""

    CKPT = ROOT / "runs/mstcn-20260817-170254/checkpoints/best.pt"
    DATASET = ROOT / "datasets/cleansight-ActionMixed-auto"
    IMAGES = ROOT / "datasets/cleansight-ActionMixed/images/train"

    @pytest.mark.skipif(
        not (CKPT.is_file() and CKPT.with_suffix(".pt.meta.json").is_file()),
        reason="缺少真实时序 checkpoint",
    )
    @pytest.mark.skipif(not DATASET.is_dir(), reason="缺少真实数据集")
    def test_full_chain_produces_three_artifacts(self, tmp_path):
        pytest.importorskip("torch")
        result = run_predict_timeline(
            self.CKPT,
            self.DATASET,
            sequence="05ba4406-clip_1781584018103_1781584033616.mp4",
            images_dir=self.IMAGES,
            out_dir=tmp_path,
            max_frames=3,
        )
        assert len(result["videos"]) == 1
        assert len(result["timelines"]) == 1
        assert len(result["band_charts"]) >= 1
        video = Path(result["videos"][0])
        timeline = Path(result["timelines"][0])
        assert video.is_file() and timeline.is_file()
        payload = __import__("json").loads(timeline.read_text(encoding="utf-8"))
        assert payload["segments"] and "frame_acc" in payload
