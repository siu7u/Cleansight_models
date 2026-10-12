"""tools/quality_report.py：自动标注质量报告测试。

验收标准（verification-first）：
- 完美对齐：presence / IoU recall/precision 全 1.0，覆盖率 100%，conf 均值正确
- 部分漏检/多余框：presence 与 IoU 的 R/P 按帧级算术正确
- enabled 覆盖率 < 90% → 健康检查告警（防 smoke 产物污染事故重演）
- 跨视频聚合计数正确；人工有标注但 auto 无产物的视频被跳过并告警
- CLI：--auto 目录/单文件、--out 写报告
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.quality_report import (  # noqa: E402
    build_quality_report,
    main,
    parse_args,
)
from framework.cleansight_eval.detection.auto_annotate import legacy_format  # noqa: E402

BOX = (20.0, 20.0, 10.0, 10.0)


def _manual_task(name: str, frames_count: int, boxes: dict) -> list[dict]:
    """人工导出 task：``boxes`` 为 ``{类别: {帧号: (x,y,w,h)}}``。"""

    results = []
    for class_name, by_frame in boxes.items():
        sequence = [
            {"frame": frame, "enabled": True, "x": box[0], "y": box[1],
             "width": box[2], "height": box[3], "time": 0.0}
            for frame, box in sorted(by_frame.items())
        ]
        results.append({
            "type": "videorectangle",
            "value": {"labels": [class_name], "framesCount": frames_count, "sequence": sequence},
        })
    return [{"id": 0, "data": {"video": name}, "annotations": [{"result": results}]}]


def _auto_task(name: str, frames_count: int, boxes: dict, confs: dict | None = None) -> list[dict]:
    """自动标注 task（经 legacy_format.build_task 构造，与 run 产出同构）。

    ``boxes`` 为 ``{类别: {帧号: (x,y,w,h)}}``；``confs`` 为 ``{类别: {帧号: conf}}``。
    """

    confs = confs or {}
    tracks = []
    for class_name, by_frame in boxes.items():
        sequence = []
        for frame in range(1, frames_count + 1):
            box = by_frame.get(frame)
            if box is None:
                sequence.append({"frame": frame, "enabled": False})
                continue
            entry = {"frame": frame, "enabled": True, "x": box[0], "y": box[1],
                     "width": box[2], "height": box[3]}
            class_conf = confs.get(class_name, {})
            if frame in class_conf:
                entry["conf"] = class_conf[frame]
            sequence.append(entry)
        tracks.append((class_name, sequence))
    return legacy_format.build_task(name, tracks, frames_count, fps=10.0, task_id=0)


class TestPerfectMatch:
    def test_all_metrics_one(self):
        manual = _manual_task("a.mp4", 3, {"hand": {1: BOX, 2: BOX, 3: BOX}})
        auto = _auto_task("a.mp4", 3, {"hand": {1: BOX, 2: BOX, 3: BOX}},
                          confs={"hand": {1: 0.9, 2: 0.8, 3: 0.7}})
        report = build_quality_report(auto, manual)
        stats = report["videos"]["a.mp4"]["classes"]["hand"]
        assert stats["presence_recall"] == 1.0
        assert stats["presence_precision"] == 1.0
        assert stats["recall_iou"] == 1.0
        assert stats["precision_iou"] == 1.0
        assert stats["mean_iou_matched"] == 1.0
        assert stats["coverage"] == 1.0
        assert stats["conf_mean"] == pytest.approx(0.8)
        assert report["warnings"] == []


class TestPartialCoverage:
    def test_missed_and_extra_frames(self):
        manual = _manual_task("a.mp4", 6, {"hand": {1: BOX, 3: BOX, 5: BOX}})
        # 自动：漏掉帧 5，多出帧 6（不同位置），帧 1/3 完美
        auto = _auto_task("a.mp4", 6, {"hand": {1: BOX, 3: BOX, 6: (60.0, 60.0, 10.0, 10.0)}})
        report = build_quality_report(auto, manual)
        stats = report["videos"]["a.mp4"]["classes"]["hand"]
        assert stats["human_frames"] == 3
        assert stats["auto_frames"] == 3
        assert stats["presence_recall"] == pytest.approx(2 / 3)
        assert stats["presence_precision"] == pytest.approx(2 / 3)
        assert stats["recall_iou"] == pytest.approx(2 / 3)
        assert stats["precision_iou"] == pytest.approx(2 / 3)
        assert stats["coverage"] == pytest.approx(3 / 6)

    def test_presence_miss_warns(self):
        # 人工 5 帧只检出 3 帧 → presence recall 0.6 < 0.9 → 漏检告警
        manual = _manual_task("a.mp4", 5, {"hand": {f: BOX for f in (1, 2, 3, 4, 5)}})
        auto = _auto_task("a.mp4", 5, {"hand": {1: BOX, 3: BOX, 5: BOX}})
        report = build_quality_report(auto, manual)
        assert any("漏检" in w and "a.mp4" in w for w in report["warnings"])

    def test_missing_class_warns(self):
        # 人工有 syringe 标注但自动产物完全没有 → 类别覆盖缺口告警
        manual = _manual_task("a.mp4", 3, {"hand": {1: BOX}, "syringe": {2: BOX}})
        auto = _auto_task("a.mp4", 3, {"hand": {1: BOX}})
        report = build_quality_report(auto, manual)
        assert any("完全没有该类别" in w and "syringe" in w for w in report["warnings"])

    def test_full_coverage_no_warning(self):
        manual = _manual_task("a.mp4", 5, {"hand": {f: BOX for f in range(1, 6)}})
        auto = _auto_task("a.mp4", 5, {"hand": {f: BOX for f in range(1, 6)}})
        report = build_quality_report(auto, manual)
        assert report["videos"]["a.mp4"]["classes"]["hand"]["coverage"] == 1.0
        assert report["warnings"] == []


class TestAggregationAndSkip:
    def test_summary_sums_across_videos(self):
        manual = [
            *_manual_task("a.mp4", 4, {"hand": {1: BOX, 2: BOX}}),
            *_manual_task("b.mp4", 4, {"hand": {1: BOX, 2: BOX, 3: BOX}}),
        ]
        auto = [
            *_auto_task("a.mp4", 4, {"hand": {1: BOX, 2: BOX}}),
            *_auto_task("b.mp4", 4, {"hand": {1: BOX, 3: BOX}}),
        ]
        report = build_quality_report(auto, manual)
        summary = report["summary"]["hand"]
        assert summary["human_frames"] == 5
        assert summary["auto_frames"] == 4
        assert summary["presence_recall"] == pytest.approx(4 / 5)  # 交集 {1,2}+{1,3}
        assert summary["presence_precision"] == pytest.approx(4 / 4)

    def test_manual_without_auto_skipped(self):
        manual = [
            *_manual_task("a.mp4", 3, {"hand": {1: BOX}}),
            *_manual_task("nope.mp4", 3, {"hand": {1: BOX}}),
        ]
        auto = _auto_task("a.mp4", 3, {"hand": {1: BOX}})
        report = build_quality_report(auto, manual)
        assert "nope.mp4" not in report["videos"]
        assert any("nope.mp4" in w for w in report["warnings"])

    def test_empty_manual_task_skipped_gracefully(self):
        # 人工导出中未标注（空 result）的 task 不应中断解析
        manual = [
            *_manual_task("a.mp4", 3, {"hand": {1: BOX}}),
            {"id": 1, "data": {"video": "b.mp4"}, "annotations": [{"result": []}]},
        ]
        auto = _auto_task("a.mp4", 3, {"hand": {1: BOX}})
        report = build_quality_report(auto, manual)
        assert "a.mp4" in report["videos"]
        assert "b.mp4" not in report["videos"]


class TestCli:
    def test_parse_args(self):
        args = parse_args(["--auto", "ann", "--manual", "m.json", "--out", "r.json", "--iou", "0.3"])
        assert args.iou == 0.3 and args.out == "r.json"

    def test_main_writes_report(self, tmp_path):
        auto_dir = tmp_path / "ann"
        auto_dir.mkdir()
        (auto_dir / "a.json").write_text(
            json.dumps(_auto_task("a.mp4", 3, {"hand": {1: BOX, 2: BOX, 3: BOX}})),
            encoding="utf-8",
        )
        manual_path = tmp_path / "manual.json"
        manual_path.write_text(
            json.dumps(_manual_task("a.mp4", 3, {"hand": {1: BOX, 2: BOX, 3: BOX}})),
            encoding="utf-8",
        )
        out = tmp_path / "report.json"
        code = main(["--auto", str(auto_dir), "--manual", str(manual_path), "--out", str(out)])
        assert code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["summary"]["hand"]["presence_recall"] == 1.0
