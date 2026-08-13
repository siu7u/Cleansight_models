"""YOLO 自动标注：legacy JSON 产出与兼容性测试。

验收标准（verification-first）：
- 轨迹划分（hand 2 条、其他 1 条）、全帧 sequence、缺席帧 enabled=false 语义
- 产出 JSON 与 legacy Label Studio 导出逐字段同构
- 产出 JSON 可被历史 ``legacy/temporal-transformer/lab.py::load_data_json``
  直接解析出 ``[T, N*5]`` 特征（核心验收：作为时序训练输入的可行性）
- 真实视频 + 真实 legacy 权重的全链路 smoke（ultralytics 可用时）
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cleansight_eval.detection import auto_annotate

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_legacy_lab():
    """动态加载历史时序代码（legacy 不被 framework import，测试用 importlib 验证兼容）。"""

    spec = importlib.util.spec_from_file_location(
        "legacy_lab", REPO_ROOT / "legacy" / "temporal-transformer" / "lab.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _det(class_name, cx, cy, w, h, conf=0.9):
    """构造一帧内的单个检测（归一化中心点）。"""

    return {"class": class_name, "confidence": conf, "xywhn": [cx, cy, w, h]}


def _mock_frame_detections():
    """5 帧 mock 检测：hand 双实例、short_brush 单实例、含缺席段。"""

    return [
        [
            _det("hand", 0.5, 0.4, 0.2, 0.1, conf=0.91),
            _det("hand", 0.3, 0.2, 0.1, 0.08, conf=0.82),  # 面积更小 → slot1
            _det("short_brush", 0.7, 0.6, 0.15, 0.12, conf=0.75),
        ],
        [_det("hand", 0.55, 0.45, 0.22, 0.11, conf=0.88)],
        [],
        [_det("short_brush", 0.65, 0.5, 0.12, 0.1, conf=0.7)],
        [],
    ]


class TestBuildTrackSequences:
    def test_track_split_and_full_coverage(self):
        tracks = auto_annotate.build_track_sequences(
            _mock_frame_detections(), frames_count=5, fps=10.0
        )
        # hand 2 条（top-2）+ short_brush 1 条 = 3 条轨迹
        assert sorted(name for name, _ in tracks) == ["hand", "hand", "short_brush"]
        for _name, sequence in tracks:
            # 全帧覆盖 [1, frames_count]，保证 legacy 消费端轨迹等长
            assert len(sequence) == 5
            assert [entry["frame"] for entry in sequence] == [1, 2, 3, 4, 5]

    def test_absent_frames_disabled_with_previous_box(self):
        tracks = auto_annotate.build_track_sequences(
            _mock_frame_detections(), frames_count=5, fps=10.0
        )
        hand_top1 = next(sequence for name, sequence in tracks if name == "hand")
        # 帧 1/2 有效，帧 3-5 缺席（外推帧 2 的框坐标）
        assert hand_top1[0]["enabled"] is True
        assert hand_top1[1]["enabled"] is True
        assert all(entry["enabled"] is False for entry in hand_top1[2:])
        last_box = hand_top1[1]
        for entry in hand_top1[2:]:
            assert entry["x"] == last_box["x"]
            assert entry["y"] == last_box["y"]
            assert "conf" not in entry  # 缺席帧无置信度

    def test_never_seen_track_zeros(self):
        # short_brush 在帧 0 出现：帧 1 缺席时外推帧 0 的框
        tracks = auto_annotate.build_track_sequences(
            _mock_frame_detections(), frames_count=5, fps=10.0
        )
        brush = next(sequence for name, sequence in tracks if name == "short_brush")
        assert brush[0]["enabled"] is True
        assert brush[1]["enabled"] is False
        assert brush[1]["x"] == brush[0]["x"]  # 缺席帧外推上一有效框
        assert brush[2]["enabled"] is False  # 帧 2 空
        assert brush[3]["enabled"] is True  # 帧 3 再次检测到
        assert brush[4]["enabled"] is False
        assert brush[4]["x"] == brush[3]["x"]

    def test_track_absent_before_first_detection_zeros(self):
        # 类别中途才出现：出现前的缺席帧从未有有效框 → 坐标全 0
        detections = [[], [], [_det("syringe", 0.5, 0.5, 0.1, 0.1)], [], []]
        tracks = auto_annotate.build_track_sequences(detections, frames_count=5, fps=10.0)
        syringe = next(sequence for name, sequence in tracks if name == "syringe")
        assert syringe[0]["enabled"] is False and syringe[0]["x"] == 0.0
        assert syringe[1]["enabled"] is False and syringe[1]["x"] == 0.0
        assert syringe[2]["enabled"] is True
        assert syringe[3]["enabled"] is False and syringe[3]["x"] == syringe[2]["x"]

    def test_coordinate_conversion_and_conf(self):
        tracks = auto_annotate.build_track_sequences(
            _mock_frame_detections(), frames_count=5, fps=10.0
        )
        hand_top1 = next(sequence for name, sequence in tracks if name == "hand")
        first = hand_top1[0]
        # cx=0.5, cy=0.4, w=0.2, h=0.1 → 左上角百分比 x=(0.5-0.1)*100=40, y=(0.4-0.05)*100=35
        assert first["x"] == pytest.approx(40.0)
        assert first["y"] == pytest.approx(35.0)
        assert first["width"] == pytest.approx(20.0)
        assert first["height"] == pytest.approx(10.0)
        assert first["conf"] == pytest.approx(0.91)
        # time=(frame-1)/fps，帧号从 1 开始
        assert first["time"] == pytest.approx(0.0)
        assert hand_top1[1]["time"] == pytest.approx(0.1)


class TestBuildTask:
    def test_legacy_structure(self):
        tracks = auto_annotate.build_track_sequences(
            _mock_frame_detections(), frames_count=5, fps=10.0
        )
        task = auto_annotate.build_task("demo.mp4", tracks, frames_count=5, fps=10.0, task_id=7)
        assert isinstance(task, list) and len(task) == 1
        item = task[0]
        assert item["id"] == 7
        assert item["data"]["video"] == "demo.mp4"
        results = item["annotations"][0]["result"]
        assert [r["type"] for r in results] == ["videorectangle"] * 3
        value = results[0]["value"]
        assert value["labels"] == ["hand"]
        assert value["framesCount"] == 5
        assert value["duration"] == pytest.approx(0.5)
        assert value["sequence"][0]["frame"] == 1
        # 只有 videorectangle，不产出 timelinelabels（YOLO 无法生成动作标签）
        assert "timelinelabels" not in {r["type"] for r in results}


class TestLegacyCompatibility:
    """核心验收：产出 JSON 可被历史时序代码直接消费。"""

    def test_load_data_json_parses_output(self, tmp_path):
        tracks = auto_annotate.build_track_sequences(
            _mock_frame_detections(), frames_count=5, fps=10.0
        )
        task = auto_annotate.build_task("demo.mp4", tracks, frames_count=5, fps=10.0, task_id=0)
        json_path = tmp_path / "demo.json"
        json_path.write_text(json.dumps(task), encoding="utf-8")

        legacy_lab = _load_legacy_lab()
        features, truths = legacy_lab.load_data_json(str(json_path), id=0)
        # 3 条轨迹 × 5 维 = 15 维，T=5 帧
        assert features.shape == (5, 15)
        assert truths == ["Idle"] * 5
        # 与 legacy 特征口径一致：有效帧特征非零
        assert np.all(features[0, :5] > 0)


class TestRunAutoAnnotateSmoke:
    """全链路 smoke：真实视频 + 真实 legacy 权重（ultralytics 可用时）。"""

    @pytest.fixture
    def _ultralytics_output_dir(self, monkeypatch, tmp_path):
        """ultralytics 8.3 检测到 pytest 运行时会写 site-packages/tests/tmp（只读）；改为临时目录。"""

        import ultralytics.cfg as ucfg

        monkeypatch.setattr(ucfg, "TESTS_RUNNING", False)
        monkeypatch.setattr(ucfg, "RUNS_DIR", tmp_path)

    @pytest.mark.skipif(
        not (REPO_ROOT / "legacy/yolo-detection/pipeline/versioned_weights/yolo-large-v3/best.pt").is_file(),
        reason="缺少 legacy 权重",
    )
    def test_smoke_with_real_video(self, tmp_path, _ultralytics_output_dir):
        ultralytics = pytest.importorskip("ultralytics")
        video = REPO_ROOT / "legacy/yolo-detection/pipeline/raw/videos"
        videos = sorted(video.glob("*.mp4"))
        if not videos:
            pytest.skip("缺少真实测试视频")
        ckpt = REPO_ROOT / "legacy/yolo-detection/pipeline/versioned_weights/yolo-large-v3/best.pt"
        outputs = auto_annotate.run_auto_annotate(
            [videos[0]],
            [{"path": ckpt, "class_map": {0: "hand", 1: "scope_control_body", 2: "scope_mid_section"}}],
            tmp_path,
            imgsz=640,
            conf=0.25,
            max_frames=30,
        )
        assert len(outputs) == 1
        task = json.loads(outputs[0].read_text(encoding="utf-8"))
        results = task[0]["annotations"][0]["result"]
        assert all(r["type"] == "videorectangle" for r in results)
        if results:  # 检测到目标时验证 legacy 可消费
            legacy_lab = _load_legacy_lab()
            features, truths = legacy_lab.load_data_json(str(outputs[0]), id=0)
            total = results[0]["value"]["framesCount"]
            assert features.shape == (total, len(results) * 5)
            assert len(truths) == total
