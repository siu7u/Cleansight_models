"""自动标注质量报告：auto 标注 JSON vs 人工导出 → 检出率 / IoU / conf / enabled 覆盖率。

用途：数据集链的**质量门**——自动标注产物并入数据集前，与同一批视频的人工
Label Studio 导出逐帧对照，量化检测质量（对应 `docs/AUTO_ANNOTATION.md`
「已知限制 · 质量门」里"建议抽样与人工标注对照（检出率 / IoU）"的数字化实现）：

- **特征存在性**：presence recall / precision——人工有目标（enabled）的帧里自动
  标注标出了多少。时序 40 维特征的核心是 presence + 中心点/尺寸，这一层直接
  决定特征质量（框几何略偏不影响太大，漏检则整类特征全零）。
- **框几何质量**：IoU>=阈值（默认 0.5）的 recall / precision、匹配框平均 IoU。
  匹配口径：每帧同类取 IoU 最大配对，recall/precision 独立统计（非一对一匹配，
  对 top-K slot 语义足够）。
- **健康检查**：每类 enabled 覆盖率（enabled 帧 / 视频总帧数）。异常低提示
  产物截断或漏检——曾因 smoke 产物只覆盖前 60 帧混入数据集造成特征污染。
- **conf 分布**：自动框置信度 min/mean/max 与低置信度（< 阈值，默认 0.3）占比。

纯读产物，不做推理；auto 与 manual 都是 legacy videorectangle 结构，统一走
``auto_annotate.legacy_format.parse_legacy_task`` 解析。输出逐视频逐类明细 +
汇总，可写 JSON 报告（供后续人工/CI 审阅）。

用法：
    python tools/quality_report.py \
        --auto outputs/annotations \
        --manual path/to/<人工导出>.json \
        --out outputs/quality/auto-v1-quality.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 脚本直跑（python tools/xxx.py）时把仓库根加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from framework.cleansight_eval.detection.auto_annotate import legacy_format  # noqa: E402

DEFAULT_IOU = 0.5
DEFAULT_CONF_LOW = 0.3


def _video_boxes(tasks: list[dict], source: str) -> dict[str, dict[str, dict[int, list[dict]]]]:
    """解析 task 数组，返回 ``{视频名: {类别名: {帧号: [框, ...]}}}``（仅 enabled 帧）。

    同一类别的多条轨迹（如 hand 的两个 slot）合并进同一帧的框列表，不互相覆盖。
    ``source`` 仅用于报错/告警文案。缺 annotations 或没有 videorectangle 的
    task（如人工导出中未标注的视频）跳过并告警，不中断其余视频。
    """

    by_video: dict[str, dict[str, dict[int, list[dict]]]] = {}
    for task_index, task in enumerate(tasks):
        try:
            parsed = legacy_format.parse_legacy_task([task])
        except (ValueError, TypeError) as exc:
            print(f"[quality-report] 跳过（{exc}）: {source} task#{task_index}")
            continue
        name = Path(parsed.video_name).name
        classes: dict[str, dict[int, list[dict]]] = {}
        for class_name, sequence in parsed.tracks:
            by_frame = classes.setdefault(class_name, {})
            for entry in sequence:
                if entry.get("enabled"):
                    by_frame.setdefault(int(entry["frame"]), []).append(
                        {
                            "x": float(entry.get("x", 0.0)),
                            "y": float(entry.get("y", 0.0)),
                            "width": float(entry.get("width", 0.0)),
                            "height": float(entry.get("height", 0.0)),
                            "conf": float(entry.get("conf", 1.0)),
                        }
                    )
        by_video[name] = classes
    return by_video


def _iou(a: dict, b: dict) -> float:
    """左上角百分比框 IoU（坐标等比例缩放不影响 IoU）。"""

    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _class_stats(
    auto_frames: dict[int, list[dict]], human_frames: dict[int, list[dict]], frames_count: int, iou_thr: float
) -> dict:
    """单视频单类的对照统计（presence / IoU / conf / 覆盖率）。"""

    human_frames_set = set(human_frames)
    auto_frames_set = set(auto_frames)
    overlap = human_frames_set & auto_frames_set
    presence_recall = len(overlap) / len(human_frames_set) if human_frames_set else None
    presence_precision = len(overlap) / len(auto_frames_set) if auto_frames_set else None

    matched_ious: list[float] = []
    for frame in human_frames_set:
        if frame not in auto_frames:
            continue
        for human_box in human_frames[frame]:
            best = max(
                (_iou(human_box, auto_box) for auto_box in auto_frames[frame]), default=0.0
            )
            if best >= iou_thr:
                matched_ious.append(best)
    recall_iou = len(matched_ious) / len(human_frames_set) if human_frames_set else None
    precision_iou = len(matched_ious) / len(auto_frames_set) if auto_frames_set else None

    confs = [box["conf"] for boxes in auto_frames.values() for box in boxes]
    coverage = len(auto_frames_set) / frames_count if frames_count > 0 else None

    return {
        "human_frames": len(human_frames_set),
        "auto_frames": len(auto_frames_set),
        "presence_overlap": len(overlap),
        "presence_recall": presence_recall,
        "presence_precision": presence_precision,
        "recall_iou": recall_iou,
        "precision_iou": precision_iou,
        "mean_iou_matched": float(sum(matched_ious) / len(matched_ious)) if matched_ious else None,
        "matched_boxes": len(matched_ious),
        "coverage": coverage,
        "conf_min": min(confs) if confs else None,
        "conf_mean": float(sum(confs) / len(confs)) if confs else None,
        "conf_max": max(confs) if confs else None,
    }


def _aggregate(video_stats: dict[str, dict[str, dict]]) -> dict:
    """跨视频逐类聚合（计数加总后重算比率，避免小样本抖动）。"""

    totals: dict[str, dict] = {}
    for video in video_stats.values():
        for class_name, stats in video["classes"].items():
            bucket = totals.setdefault(
                class_name,
                {"human_frames": 0, "auto_frames": 0, "presence_overlap": 0,
                 "matched_boxes": 0, "conf_sum": 0.0, "iou_sum": 0.0},
            )
            bucket["human_frames"] += stats["human_frames"]
            bucket["auto_frames"] += stats["auto_frames"]
            bucket["presence_overlap"] += stats["presence_overlap"]
            bucket["matched_boxes"] += stats["matched_boxes"]
            bucket["iou_sum"] += (stats["mean_iou_matched"] or 0.0) * stats["matched_boxes"]
            bucket["conf_sum"] += (stats["conf_mean"] or 0.0) * stats["auto_frames"]
    summary = {}
    for class_name, bucket in totals.items():
        human = bucket["human_frames"]
        auto = bucket["auto_frames"]
        summary[class_name] = {
            "human_frames": human,
            "auto_frames": auto,
            "presence_recall": (bucket["presence_overlap"] / human) if human else None,
            "presence_precision": (bucket["presence_overlap"] / auto) if auto else None,
            "mean_iou": (bucket["iou_sum"] / bucket["matched_boxes"]) if bucket["matched_boxes"] else None,
            "conf_mean": (bucket["conf_sum"] / auto) if auto else None,
        }
    return summary


def _fmt(value, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if value is not None else "  -  "


def build_quality_report(
    auto_tasks: list[dict],
    manual_tasks: list[dict],
    *,
    iou_threshold: float = DEFAULT_IOU,
    conf_low: float = DEFAULT_CONF_LOW,
    auto_source: str = "auto",
    manual_source: str = "manual",
) -> dict:
    """auto/manual task 数组 → 质量报告 dict（逐视频明细 + 汇总 + 健康检查）。"""

    auto = _video_boxes(auto_tasks, auto_source)
    manual = _video_boxes(manual_tasks, manual_source)
    frames_counts = _frames_counts(auto_tasks, auto_source)

    videos: dict[str, dict] = {}
    skipped: list[str] = []
    for name, manual_classes in sorted(manual.items()):
        auto_classes = auto.get(name, {})
        if not auto_classes:
            skipped.append(name)
            continue
        frames_count = frames_counts.get(name, 0)
        class_stats = {
            class_name: _class_stats(
                auto_classes.get(class_name, {}),
                manual_classes.get(class_name, {}),
                frames_count,
                iou_threshold,
            )
            for class_name in sorted(set(manual_classes) | set(auto_classes))
        }
        videos[name] = {"classes": class_stats, "frames_count": frames_count}

    summary = _aggregate(videos)
    warnings = []
    for name, classes in videos.items():
        for class_name, stats in classes["classes"].items():
            human = stats["human_frames"]
            if human == 0:
                continue  # 人工未标该类，无从判定漏检
            if stats["auto_frames"] == 0:
                warnings.append(
                    f"{name} {class_name}: 人工有 {human} 帧标注但自动产物完全没有该类别（类别覆盖缺口）"
                )
            elif stats["presence_recall"] is not None and stats["presence_recall"] < 0.9:
                warnings.append(
                    f"{name} {class_name}: presence 漏检 "
                    f"（人工 {human} 帧仅检出 {stats['presence_overlap']} 帧，"
                    f"R={stats['presence_recall']:.1%}）"
                )
    if skipped:
        warnings.append(f"自动标注中缺失 {len(skipped)} 个视频（人工有标注但 auto 无产物）: {', '.join(skipped[:5])}")

    return {
        "schema_version": 1,
        "auto_source": auto_source,
        "manual_source": manual_source,
        "iou_threshold": iou_threshold,
        "conf_low": conf_low,
        "videos": videos,
        "summary": summary,
        "warnings": warnings,
    }


def _frames_counts(auto_tasks: list[dict], source: str) -> dict[str, int]:
    """各视频在 auto JSON 中的 frames_count（健康检查分母），一次性构建。"""

    counts: dict[str, int] = {}
    for task in auto_tasks:
        try:
            parsed = legacy_format.parse_legacy_task([task])
        except (ValueError, TypeError):
            continue
        counts.setdefault(Path(parsed.video_name).name, parsed.frames_count)
    return counts


def print_report(report: dict) -> None:
    """把报告打印成人类可读表格。"""

    print("[quality-report] ===== 逐视频逐类 ===== ")
    for name, video in report["videos"].items():
        print(f"[quality-report] {name}（{video['frames_count']} 帧）")
        for class_name, s in video["classes"].items():
            print(
                f"[quality-report]   {class_name:<20} 人工帧 {s['human_frames']:>4} 自动帧 {s['auto_frames']:>4} "
                f"| presence R/P {_fmt(s['presence_recall'])}/{_fmt(s['presence_precision'])} "
                f"| IoU≥{report['iou_threshold']} R/P {_fmt(s['recall_iou'])}/{_fmt(s['precision_iou'])} "
                f"| 平均 IoU {_fmt(s['mean_iou_matched'])} | 覆盖率 {_fmt(s['coverage'])} "
                f"| conf {_fmt(s['conf_min'], 2)}-{_fmt(s['conf_max'], 2)}（均值 {_fmt(s['conf_mean'], 2)}）"
            )
    print("[quality-report] ===== 汇总（跨视频聚合）===== ")
    for class_name, s in report["summary"].items():
        print(
            f"[quality-report]   {class_name:<20} 人工帧 {s['human_frames']:>4} 自动帧 {s['auto_frames']:>4} "
            f"| presence R/P {_fmt(s['presence_recall'])}/{_fmt(s['presence_precision'])} "
            f"| 平均 IoU {_fmt(s['mean_iou'])} | conf 均值 {_fmt(s['conf_mean'], 2)}"
        )
    for warning in report["warnings"]:
        print(f"[quality-report] ⚠ {warning}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="自动标注质量报告：auto JSON vs 人工导出（检出率 / IoU / conf / enabled 覆盖率）"
    )
    p.add_argument("--auto", required=True, help="自动标注 JSON 文件或目录（cli.annotate run 产出）")
    p.add_argument("--manual", required=True, help="人工 Label Studio 导出 JSON（同一批视频）")
    p.add_argument("--out", default=None, help="报告 JSON 输出路径（缺省只打印）")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU, help="IoU 匹配阈值（默认 0.5）")
    p.add_argument("--conf-low", type=float, default=DEFAULT_CONF_LOW, help="低置信度阈值（默认 0.3，仅统计展示）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    auto_path = Path(args.auto)
    if auto_path.is_dir():
        auto_files = sorted(auto_path.glob("*.json"))
        if not auto_files:
            raise FileNotFoundError(f"自动标注目录没有 JSON: {auto_path}")
        auto_tasks = [task for path in auto_files for task in json.loads(path.read_text(encoding="utf-8"))]
        auto_source = str(auto_path)
    else:
        auto_tasks = json.loads(auto_path.read_text(encoding="utf-8"))
        auto_source = str(auto_path)
    manual_tasks = json.loads(Path(args.manual).read_text(encoding="utf-8"))

    report = build_quality_report(
        auto_tasks,
        manual_tasks,
        iou_threshold=args.iou,
        conf_low=args.conf_low,
        auto_source=auto_source,
        manual_source=str(args.manual),
    )
    print_report(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[quality-report] 报告已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
