"""YOLO 自动标注：legacy JSON 产出与兼容性测试。

验收标准（verification-first）：
- 轨迹划分（hand 2 条、其他 1 条）、全帧 sequence、缺席帧 enabled=false 语义
- 产出 JSON 与 legacy Label Studio 导出逐字段同构
- 产出 JSON 可被历史 ``legacy/temporal-transformer/lab.py::load_data_json``
  直接解析出 ``[T, N*5]`` 特征（核心验收：作为时序训练输入的可行性）
- 优化项：批量推理、帧采样复用、类别级置信度、ByteTrack 轨迹、断点续跑
- run-dataset：图片帧序列数据集 → 时序训练数据（标签帧对齐、8 类编号、load_split 消费验收）
- 真实视频 + 真实 legacy 权重的全链路 smoke（ultralytics 可用时）
"""

import importlib.util
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from cleansight_eval.detection import auto_annotate
from cleansight_eval.detection import inference

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_legacy_lab():
    """动态加载历史时序代码（legacy 不被 framework import，测试用 importlib 验证兼容）。"""

    spec = importlib.util.spec_from_file_location(
        "legacy_lab", REPO_ROOT / "legacy" / "temporal-transformer" / "lab.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def _ultralytics_output_dir(monkeypatch, tmp_path):
    """把 ultralytics 中间产物目录改为临时目录（8.4+ 已由 settings 管理，此处仅兼容 8.3）。"""

    import ultralytics.cfg as ucfg

    if hasattr(ucfg, "RUNS_DIR"):  # 8.4+ 已移除 cfg.RUNS_DIR
        monkeypatch.setattr(ucfg, "RUNS_DIR", tmp_path)
    if hasattr(ucfg, "TESTS_RUNNING"):
        monkeypatch.setattr(ucfg, "TESTS_RUNNING", False)


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


class TestRunDatasetAnnotate:
    """run-dataset：图片帧序列数据集 → 时序训练数据（mock 推理，不依赖 ultralytics）。

    验收标准（verification-first）：
    - 输入 images/<split>/<序列>-<帧号:06d>.jpg 有序帧 + labels/<split>/<序列>.txt
      动作标签（"frame_id action_id"）
    - 只为有动作标签的帧产出 frames/<split>/<序列>-<帧号:06d>.txt（8 类全局编号）
    - 动作标签复制到输出根；labels/frames data.yaml 缺省补写（已有不覆盖）
    - 断点续跑跳过已完成序列/帧；缺标签、标签帧无图、帧号缺失明确报错
    - 产出可被 temporal/data.py::load_split 直接消费（核心验收）
    """

    @staticmethod
    def _make_dataset(
        root: Path, seq: str = "demo.mp4", frames=(1, 5, 9, 13), split: str = "train", actions=(0, 1, 2, 3)
    ) -> None:
        """构造一个 split 的图片帧序列 + 动作标签。"""

        for frame, _action in zip(frames, actions):
            image_path = root / "images" / split / f"{seq}-{frame:06d}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, dtype=np.uint8))
        label_path = root / "labels" / split / f"{seq}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(f"{f} {a}" for f, a in zip(frames, actions)) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _fake_predict(model, frames, **kwargs):
        """每帧返回同一组检测（class_id 为本地 id，经 class_map 映射为全局类名）。"""

        detections = [
            {"class_id": 0, "confidence": 0.9, "xywhn": [0.5, 0.5, 0.1, 0.1]},   # hand → 8 类 id 0
            {"class_id": 1, "confidence": 0.8, "xywhn": [0.25, 0.25, 0.2, 0.2]},  # syringe → 8 类 id 4
        ]
        return [detections for _ in frames]

    @staticmethod
    def _patch_inference(monkeypatch):
        """替换模型加载与推理（dataset 经 run._load_models / _infer_batch 调用）。"""

        monkeypatch.setattr(
            inference, "load_predictor", lambda *a, **k: (object(), {0: "hand", 1: "syringe"})
        )

    @staticmethod
    def _fake_ckpt(tmp_path) -> Path:
        ckpt = tmp_path / "fake.pt"
        ckpt.write_bytes(b"fake-ckpt")
        return ckpt

    def test_writes_frames_for_label_frames_only(self, tmp_path, monkeypatch):
        root = tmp_path / "ds"
        self._make_dataset(root)
        self._patch_inference(monkeypatch)
        monkeypatch.setattr(inference, "predict_frames", self._fake_predict)

        auto_annotate.run_dataset_annotate(
            root,
            [{"path": self._fake_ckpt(tmp_path), "class_map": {0: "hand", 1: "syringe"}}],
            tmp_path / "out",
            batch_size=2,
        )
        # 只为标签帧（1/5/9/13）产出 frames；行格式 8 类编号 + 6 位小数归一化
        for frame in (1, 5, 9, 13):
            frame_txt = tmp_path / "out" / "frames" / "train" / f"demo.mp4-{frame:06d}.txt"
            assert frame_txt.is_file()
            assert frame_txt.read_text(encoding="utf-8") == (
                "0 0.500000 0.500000 0.100000 0.100000\n"
                "4 0.250000 0.250000 0.200000 0.200000\n"
            )
        assert not (tmp_path / "out" / "frames" / "train" / "demo.mp4-000002.txt").exists()

    def test_labels_copied_and_data_yamls_written(self, tmp_path, monkeypatch):
        root = tmp_path / "ds"
        self._make_dataset(root)
        self._patch_inference(monkeypatch)
        monkeypatch.setattr(inference, "predict_frames", self._fake_predict)
        specs = [{"path": self._fake_ckpt(tmp_path), "class_map": {0: "hand", 1: "syringe"}}]
        out = tmp_path / "out"

        auto_annotate.run_dataset_annotate(root, specs, out)
        copied = out / "labels" / "train" / "demo.mp4.txt"
        assert copied.read_text(encoding="utf-8") == (
            root / "labels" / "train" / "demo.mp4.txt"
        ).read_text(encoding="utf-8")
        labels_yaml = yaml.safe_load((out / "labels" / "data.yaml").read_text(encoding="utf-8"))
        frames_yaml = yaml.safe_load((out / "frames" / "data.yaml").read_text(encoding="utf-8"))
        assert labels_yaml["nc"] == 6 and frames_yaml["nc"] == 8

        # 已有 data.yaml 不覆盖
        out2 = tmp_path / "out2"
        (out2 / "labels").mkdir(parents=True)
        (out2 / "labels" / "data.yaml").write_text("nc: 99\n", encoding="utf-8")
        auto_annotate.run_dataset_annotate(root, specs, out2)
        assert (out2 / "labels" / "data.yaml").read_text(encoding="utf-8") == "nc: 99\n"

    def test_detections_outside_8_class_table_dropped(self, tmp_path, monkeypatch):
        root = tmp_path / "ds"
        self._make_dataset(root)
        self._patch_inference(monkeypatch)

        def fake_predict(model, frames, **kwargs):
            return [
                [
                    {"class_id": 0, "confidence": 0.9, "xywhn": [0.5] * 4},   # hand → 保留
                    {"class_id": 1, "confidence": 0.8, "xywhn": [0.3] * 4},  # 映射为表外类别 → 丢弃
                ]
                for _ in frames
            ]

        monkeypatch.setattr(inference, "predict_frames", fake_predict)
        auto_annotate.run_dataset_annotate(
            root,
            [{"path": self._fake_ckpt(tmp_path), "class_map": {0: "hand", 1: "not_in_table"}}],
            tmp_path / "out",
        )
        frame_txt = tmp_path / "out" / "frames" / "train" / "demo.mp4-000001.txt"
        assert frame_txt.read_text(encoding="utf-8") == "0 0.500000 0.500000 0.500000 0.500000\n"

    def test_class_conf_filtering(self, tmp_path, monkeypatch):
        root = tmp_path / "ds"
        self._make_dataset(root)
        self._patch_inference(monkeypatch)

        def fake_predict(model, frames, **kwargs):
            return [
                [
                    {"class_id": 0, "confidence": 0.3, "xywhn": [0.5] * 4},  # hand 低于阈值 → 丢弃
                    {"class_id": 1, "confidence": 0.9, "xywhn": [0.3] * 4},
                ]
                for _ in frames
            ]

        monkeypatch.setattr(inference, "predict_frames", fake_predict)
        auto_annotate.run_dataset_annotate(
            root,
            [{"path": self._fake_ckpt(tmp_path), "class_map": {0: "hand", 1: "syringe"}}],
            tmp_path / "out",
            conf={"hand": 0.5, "syringe": 0.5},
        )
        frame_txt = tmp_path / "out" / "frames" / "train" / "demo.mp4-000001.txt"
        assert frame_txt.read_text(encoding="utf-8") == "4 0.300000 0.300000 0.300000 0.300000\n"

    def test_resume_skips_completed_seq_and_frames(self, tmp_path, monkeypatch):
        root = tmp_path / "ds"
        self._make_dataset(root)
        self._patch_inference(monkeypatch)
        predict_calls = {"count": 0}

        def counting_predict(model, frames, **kwargs):
            predict_calls["count"] += len(frames)
            return self._fake_predict(model, frames)

        monkeypatch.setattr(inference, "predict_frames", counting_predict)
        specs = [{"path": self._fake_ckpt(tmp_path), "class_map": {0: "hand", 1: "syringe"}}]
        out = tmp_path / "out"
        auto_annotate.run_dataset_annotate(root, specs, out)
        assert predict_calls["count"] == 4
        # 完整序列已存在 → resume 后不推理；删掉一帧再跑 → 只补该帧
        auto_annotate.run_dataset_annotate(root, specs, out, resume=True)
        assert predict_calls["count"] == 4
        (out / "frames" / "train" / "demo.mp4-000013.txt").unlink()
        auto_annotate.run_dataset_annotate(root, specs, out, resume=True)
        assert predict_calls["count"] == 5

    def test_inplace_out_root_skips_label_self_copy(self, tmp_path, monkeypatch):
        """默认原地输出（--out = 数据集根）时标签自复制应跳过而不是报 SameFileError。"""

        root = tmp_path / "ds"
        self._make_dataset(root)
        self._patch_inference(monkeypatch)
        monkeypatch.setattr(inference, "predict_frames", self._fake_predict)
        auto_annotate.run_dataset_annotate(
            root,
            [{"path": self._fake_ckpt(tmp_path), "class_map": {0: "hand", 1: "syringe"}}],
            root,  # 原地
        )
        assert (root / "frames" / "train" / "demo.mp4-000001.txt").is_file()
        assert (root / "labels" / "train" / "demo.mp4.txt").is_file()

    def test_missing_labels_error(self, tmp_path, monkeypatch):
        root = tmp_path / "ds"
        self._make_dataset(root)
        (root / "labels" / "train" / "demo.mp4.txt").unlink()
        self._patch_inference(monkeypatch)
        monkeypatch.setattr(inference, "predict_frames", self._fake_predict)
        with pytest.raises(FileNotFoundError, match="缺少动作标签"):
            auto_annotate.run_dataset_annotate(
                root,
                [{"path": self._fake_ckpt(tmp_path), "class_map": {0: "hand"}}],
                tmp_path / "out",
            )

    def test_label_frame_without_image_error(self, tmp_path, monkeypatch):
        root = tmp_path / "ds"
        self._make_dataset(root, frames=(1, 5))
        (root / "labels" / "train" / "demo.mp4.txt").write_text("1 0\n99 0\n", encoding="utf-8")
        self._patch_inference(monkeypatch)
        monkeypatch.setattr(inference, "predict_frames", self._fake_predict)
        with pytest.raises(FileNotFoundError, match="标签帧 99"):
            auto_annotate.run_dataset_annotate(
                root,
                [{"path": self._fake_ckpt(tmp_path), "class_map": {0: "hand"}}],
                tmp_path / "out",
            )

    def test_missing_images_dir_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="缺少 images/"):
            auto_annotate.run_dataset_annotate(
                tmp_path / "nope",
                [{"path": tmp_path / "fake.pt", "class_map": {0: "hand"}}],
                tmp_path / "out",
            )

    def test_invalid_frame_suffix_error(self, tmp_path, monkeypatch):
        root = tmp_path / "ds"
        image_path = root / "images" / "train" / "demo.jpg"  # 无帧号后缀
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), np.full((64, 64, 3), 128, dtype=np.uint8))
        label_path = root / "labels" / "train" / "demo.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("1 0\n", encoding="utf-8")
        self._patch_inference(monkeypatch)
        monkeypatch.setattr(inference, "predict_frames", self._fake_predict)
        with pytest.raises(ValueError, match="缺少 '-<帧号:06d>' 后缀"):
            auto_annotate.run_dataset_annotate(
                root,
                [{"path": self._fake_ckpt(tmp_path), "class_map": {0: "hand"}}],
                tmp_path / "out",
            )

    def test_output_consumable_by_temporal_load_split(self, tmp_path, monkeypatch):
        """核心验收：产出可直接被 temporal/data.py::load_split 消费（40 维特征）。"""

        from cleansight_eval.temporal import data as temporal_data

        root = tmp_path / "ds"
        self._make_dataset(root)
        self._patch_inference(monkeypatch)
        monkeypatch.setattr(inference, "predict_frames", self._fake_predict)
        out = tmp_path / "out"
        auto_annotate.run_dataset_annotate(
            root,
            [{"path": self._fake_ckpt(tmp_path), "class_map": {0: "hand", 1: "syringe"}}],
            out,
        )
        features, truths, id2name = temporal_data.load_split(
            {"root": str(out), "labels_dir": "labels", "frames_dir": "frames", "fps": 7.5},
            split="train",
        )
        assert len(features) == 1 and len(truths) == 1
        assert features[0].shape == (4, 40)  # 4 个标签帧 × 8 类 × 5 维
        assert truths[0].tolist() == [0, 1, 2, 3]
        assert list(id2name.values()) == auto_annotate.ACTION_CLASSES

    @pytest.mark.skipif(
        not (REPO_ROOT / "legacy/yolo-detection/pipeline/versioned_weights/yolo-large-v3/best.pt").is_file(),
        reason="缺少 legacy 权重",
    )
    def test_smoke_with_real_dataset(self, tmp_path, _ultralytics_output_dir):
        """真实图片帧 + 真实 legacy 权重的全链路 smoke（ultralytics 可用时）。"""

        pytest.importorskip("ultralytics")
        src_images = REPO_ROOT / "datasets/cleansight-ActionMixed/images/train"
        src_labels = REPO_ROOT / "datasets/cleansight-ActionMixed/labels/train"
        if not src_labels.is_dir():
            pytest.skip("缺少真实时序数据集")
        root = tmp_path / "ds"
        seq = sorted(src_labels.glob("*.txt"))[0].stem
        label_lines = [
            line for line in (src_labels / f"{seq}.txt").read_text(encoding="utf-8").splitlines()
            if line
        ][:10]
        frames = [int(line.split()[0]) for line in label_lines]
        for frame in frames:
            image = src_images / f"{seq}-{frame:06d}.jpg"
            if not image.is_file():
                pytest.skip(f"缺少真实图片帧: {image.name}")
            target = root / "images" / "train" / image.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
        label_target = root / "labels" / "train" / f"{seq}.txt"
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        ckpt = REPO_ROOT / "legacy/yolo-detection/pipeline/versioned_weights/yolo-large-v3/best.pt"
        outputs = auto_annotate.run_dataset_annotate(
            root,
            [{"path": ckpt, "class_map": {0: "hand", 1: "scope_control_body", 2: "scope_mid_section"}}],
            tmp_path / "out",
            imgsz=640,
            conf=0.25,
            batch_size=4,
        )
        for frame in frames:
            assert (tmp_path / "out" / "frames" / "train" / f"{seq}-{frame:06d}.txt").is_file()
        assert (tmp_path / "out" / "labels" / "train" / f"{seq}.txt").is_file()
        assert len(outputs) >= len(frames)


class TestRunDatasetCli:
    """run-dataset CLI：--dataset/--out 路径与参数传递（mock 核心函数）。"""

    @staticmethod
    def _make_config(tmp_path) -> Path:
        config = tmp_path / "cfg.yaml"
        config.write_text(
            "checkpoints:\n  - path: fake.pt\n    class_map:\n      0: hand\n",
            encoding="utf-8",
        )
        return config

    @staticmethod
    def _run(monkeypatch, argv: list[str]):
        """用真实 argparse 解析并执行 handler，捕获传给 run_dataset_annotate 的参数。"""

        from cleansight_eval.cli import annotate as cli

        captured: dict = {}
        monkeypatch.setattr(
            auto_annotate,
            "run_dataset_annotate",
            lambda dataset_root, specs, out_root, **kwargs: captured.update(
                dataset_root=Path(dataset_root), out_root=Path(out_root), kwargs=kwargs
            )
            or [],
        )
        args = cli._build_parser().parse_args(argv)
        return args.handler(args), captured

    def test_dataset_root_and_out_default(self, tmp_path, monkeypatch):
        config = self._make_config(tmp_path)
        code, captured = self._run(
            monkeypatch,
            ["run-dataset", "--dataset", str(tmp_path / "ds"), "--config", str(config)],
        )
        assert code == 0
        assert captured["dataset_root"] == tmp_path / "ds"
        assert captured["out_root"] == tmp_path / "ds"  # 默认原地补写 frames/

    def test_out_override(self, tmp_path, monkeypatch):
        config = self._make_config(tmp_path)
        code, captured = self._run(
            monkeypatch,
            [
                "run-dataset",
                "--dataset", str(tmp_path / "ds"),
                "--out", str(tmp_path / "out2"),
                "--config", str(config),
            ],
        )
        assert code == 0
        assert captured["out_root"] == tmp_path / "out2"

    def test_dataset_required(self, tmp_path):
        from cleansight_eval.cli import annotate as cli

        config = self._make_config(tmp_path)
        with pytest.raises(SystemExit):  # argparse required 参数缺失
            cli._build_parser().parse_args(["run-dataset", "--config", str(config)])

    def test_conf_and_resume_passed_through(self, tmp_path, monkeypatch):
        config = self._make_config(tmp_path)
        code, captured = self._run(
            monkeypatch,
            [
                "run-dataset",
                "--dataset", str(tmp_path / "ds"),
                "--config", str(config),
                "--conf", "0.3", "--resume", "--imgsz", "320", "--batch-size", "4",
            ],
        )
        assert code == 0
        assert captured["kwargs"]["conf"] == 0.3
        assert captured["kwargs"]["resume"] is True
        assert captured["kwargs"]["imgsz"] == 320
        assert captured["kwargs"]["batch_size"] == 4

class TestConvertSkipsUnlabeled:
    """convert：无人工标签的视频跳过并告警，不中断其余视频的转换。"""

    @staticmethod
    def _auto_json(annotation_dir: Path, video_name: str, task_id: int) -> None:
        detections = [
            [{"class": "hand", "confidence": 0.9, "xywhn": [0.5, 0.5, 0.1, 0.1]}]
            for _ in range(3)
        ]
        tracks = auto_annotate.build_track_sequences(detections, frames_count=3, fps=10.0)
        task = auto_annotate.build_task(video_name, tracks, frames_count=3, fps=10.0, task_id=task_id)
        (annotation_dir / f"{video_name}.json").write_text(json.dumps(task), encoding="utf-8")

    @staticmethod
    def _manual_export(tmp_path, with_box_videos=(), empty_videos=()):
        """构造人工导出：with_box_videos 有框+动作标签；empty_videos 只有空 result。"""

        tasks = []
        for task_id, name in enumerate(with_box_videos):
            tasks.append(
                {
                    "id": task_id,
                    "data": {"video": name},
                    "annotations": [
                        {
                            "result": [
                                {
                                    "type": "videorectangle",
                                    "value": {
                                        "labels": ["hand"],
                                        "framesCount": 120,
                                        "duration": 5.0,
                                        "sequence": [],
                                    },
                                },
                                {
                                    "type": "timelinelabels",
                                    "value": {
                                        "timelinelabels": ["flush"],
                                        "ranges": [{"start": 1, "end": 100}],
                                    },
                                },
                            ]
                        }
                    ],
                }
            )
        for task_id, name in enumerate(empty_videos, start=len(with_box_videos)):
            tasks.append({"id": task_id, "data": {"video": name}, "annotations": [{"result": []}]})
        export = tmp_path / "manual.json"
        export.write_text(json.dumps(tasks), encoding="utf-8")
        return export

    def test_missing_and_empty_manual_skipped(self, tmp_path, capsys):
        annotation_dir = tmp_path / "ann"
        annotation_dir.mkdir()
        self._auto_json(annotation_dir, "a.mp4", task_id=0)  # 有标签 → 转换
        self._auto_json(annotation_dir, "b.mp4", task_id=1)  # 导出中缺失 → 跳过
        self._auto_json(annotation_dir, "c.mp4", task_id=2)  # 导出中空 task → 跳过
        manual = self._manual_export(tmp_path, with_box_videos=("a.mp4",), empty_videos=("c.mp4",))
        out = tmp_path / "out"

        outputs = auto_annotate.convert_annotations(annotation_dir, manual, out, split="train")
        assert len(outputs) == 1  # 只有 a.mp4 转换
        assert (out / "labels" / "train" / "a.mp4.txt").is_file()
        assert (out / "frames" / "train" / "a.mp4-000001.txt").is_file()
        assert not (out / "labels" / "train" / "b.mp4.txt").exists()
        assert not (out / "labels" / "train" / "c.mp4.txt").exists()
        log = capsys.readouterr().out
        assert "跳过" in log and "b.mp4" in log and "c.mp4" in log
        assert "跳过 2 个" in log  # 结尾汇总

    def test_labels_only_export_converts_with_scale_one(self, tmp_path, capsys):
        """LS 只标动作阶段（无 videorectangle）也能转换：帧号按 1:1 并告警。"""
        annotation_dir = tmp_path / "ann"
        annotation_dir.mkdir()
        self._auto_json(annotation_dir, "d.mp4", task_id=0)
        export = tmp_path / "manual_labels_only.json"
        export.write_text(
            json.dumps(
                [
                    {
                        "id": 0,
                        "data": {"video": "d.mp4"},
                        "annotations": [
                            {
                                "result": [
                                    {
                                        "type": "timelinelabels",
                                        "value": {
                                            "timelinelabels": ["flush"],
                                            "ranges": [{"start": 1, "end": 2}],
                                        },
                                    }
                                ]
                            }
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        out = tmp_path / "out2"

        outputs = auto_annotate.convert_annotations(annotation_dir, export, out, split="train")
        assert len(outputs) == 1  # 无框导出也能转换
        label_lines = (out / "labels" / "train" / "d.mp4.txt").read_text(encoding="utf-8").splitlines()
        # 1:1 换算：3 帧真实视频（10fps → stride 1），LS 区间 [1,2] 直接映射到真实帧 1..2
        assert label_lines == ["1 2", "2 2", "3 0"]  # flush=2，帧 3 未标区间 → idle=0
        log = capsys.readouterr().out
        assert "无 LS 帧率锚点" in log and "1:1 换算" in log
        assert "d.mp4" in log


class TestRunAutoAnnotateSmoke:
    """全链路 smoke：真实视频 + 真实 legacy 权重（ultralytics 可用时）。"""

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

    @pytest.mark.skipif(
        not (REPO_ROOT / "legacy/yolo-detection/pipeline/versioned_weights/yolo-large-v3/best.pt").is_file(),
        reason="缺少 legacy 权重",
    )
    def test_smoke_with_tracking_and_stride(self, tmp_path, _ultralytics_output_dir):
        """真实视频 + ByteTrack 跟踪 + 帧采样 + 批量的全链路。"""

        pytest.importorskip("ultralytics")
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
            conf={"hand": 0.2, "scope_control_body": 0.15, "scope_mid_section": 0.15},
            max_frames=40,
            frame_stride=2,
            track=True,
            batch_size=8,
        )
        assert len(outputs) == 1
        task = json.loads(outputs[0].read_text(encoding="utf-8"))
        results = task[0]["annotations"][0]["result"]
        assert all(r["type"] == "videorectangle" for r in results)


class TestDetectVideoOptimizations:
    """批量推理 / 帧采样 / 类别级置信度的行为验证（mock 推理，不依赖 ultralytics）。"""

    @staticmethod
    def _make_video(tmp_path, frames: int = 6) -> Path:
        path = tmp_path / "demo.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
        )
        for _ in range(frames):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        return path

    @staticmethod
    def _fake_detections(frames):
        return [
            [{"class_id": 0, "confidence": 0.9, "xywhn": [0.5, 0.5, 0.1, 0.1]}]
            for _ in frames
        ]

    def test_frame_stride_reuses_last_result(self, tmp_path, monkeypatch):
        video = self._make_video(tmp_path, frames=6)
        calls: list[int] = []

        def fake_predict_frames(model, frames, *, imgsz, conf, track):
            calls.append(len(frames))
            return self._fake_detections(frames)

        monkeypatch.setattr(inference, "predict_frames", fake_predict_frames)
        _total, _fps, detections = auto_annotate.detect_video(
            video, [(object(), {0: "hand"})], imgsz=640, conf=0.25,
            frame_stride=2, batch_size=4,
        )
        # 6 帧：推理帧 0/2/4 共 3 帧（一个 batch），复用帧 1/3/5
        assert sum(calls) == 3
        assert len(detections) == 6
        for index in (1, 3, 5):
            assert detections[index] == detections[index - 1]  # 复用最近推理帧结果

    def test_batch_flush_counts(self, tmp_path, monkeypatch):
        video = self._make_video(tmp_path, frames=5)

        def fake_predict_frames(model, frames, *, imgsz, conf, track):
            return self._fake_detections(frames)

        monkeypatch.setattr(inference, "predict_frames", fake_predict_frames)
        _total, _fps, detections = auto_annotate.detect_video(
            video, [(object(), {0: "hand"})], imgsz=640, conf=0.25, batch_size=2,
        )
        assert len(detections) == 5
        assert all(detections[index] == detections[0] for index in range(5))

    def test_class_conf_filtering(self, tmp_path, monkeypatch):
        video = self._make_video(tmp_path, frames=2)

        def fake_predict_frames(model, frames, *, imgsz, conf, track):
            return [
                [
                    {"class_id": 0, "confidence": 0.9, "xywhn": [0.5] * 4},  # hand，通过
                    {"class_id": 1, "confidence": 0.3, "xywhn": [0.5] * 4},  # syringe，低于阈值
                ]
                for _ in frames
            ]

        monkeypatch.setattr(inference, "predict_frames", fake_predict_frames)
        _total, _fps, detections = auto_annotate.detect_video(
            video, [(object(), {0: "hand", 1: "syringe"})], imgsz=640,
            conf={"hand": 0.5, "syringe": 0.6}, batch_size=4,
        )
        assert len(detections[0]) == 1
        assert detections[0][0]["class"] == "hand"

    def test_track_sequences_grouped_by_instance(self):
        frame_detections = [
            [
                {"class": "hand", "confidence": 0.9, "xywhn": [0.5] * 4, "track_id": 1},
                {"class": "hand", "confidence": 0.8, "xywhn": [0.3] * 4, "track_id": 2},
            ],
            [{"class": "hand", "confidence": 0.85, "xywhn": [0.52] * 4, "track_id": 1}],
            [],
        ]
        tracks = auto_annotate.build_track_sequences(
            frame_detections, frames_count=3, fps=10.0, track=True
        )
        # 按 (类别, 实例 id) 分组 → 两条 hand 轨迹，id 顺序确定
        assert sorted(name for name, _sequence in tracks) == ["hand", "hand"]
        id1, id2 = tracks  # 排序后 (hand,1) 在前
        assert id1[1][0]["enabled"] is True and id1[1][1]["enabled"] is True
        assert id1[1][2]["enabled"] is False  # 帧 2 缺席，外推
        assert id2[1][0]["enabled"] is True
        assert id2[1][1]["enabled"] is False
        assert id2[1][2]["enabled"] is False

    def test_resume_skips_existing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            auto_annotate.inference, "load_predictor", lambda *a, **k: (object(), {0: "hand"})
        )
        detect_calls = {"count": 0}

        def fake_detect(*_a, **_k):
            detect_calls["count"] += 1
            detection = {"class": "hand", "confidence": 0.9, "xywhn": [0.5, 0.5, 0.1, 0.1]}
            return (3, 10.0, [[detection]] * 3)

        monkeypatch.setattr(auto_annotate.run, "detect_video", fake_detect)
        video = tmp_path / "demo.mp4"
        video.write_bytes(b"fake-video")
        ckpt = tmp_path / "fake.pt"
        ckpt.write_bytes(b"fake-ckpt")
        out_dir = tmp_path / "out"
        specs = [{"path": ckpt, "class_map": {0: "hand"}}]

        auto_annotate.run_auto_annotate([video], specs, out_dir)
        auto_annotate.run_auto_annotate([video], specs, out_dir, resume=True)
        assert detect_calls["count"] == 1  # resume 时第二次跳过推理
        assert len(list(out_dir.glob("*.json"))) == 1
