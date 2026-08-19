"""YOLO 自动标注 CLI：python -m framework.cleansight_eval.cli.annotate。

子命令：
- ``run``：用已训练 YOLO checkpoint 对无标注视频逐帧检测，产出 legacy 时序
  标注 JSON（与 Label Studio 导出同构，可被历史 ``lab.py`` 直接消费）。
- ``convert``：自动标注 JSON + 人工 Label Studio 导出（timelinelabels 动作
  标签）→ framework 时序训练数据布局（``labels/<split>/`` + ``frames/<split>/``），
  供 ``temporal/data.py`` 直接消费。

用法（仓库根执行）:
    # run：单视频
    python -m framework.cleansight_eval.cli.annotate run \
        --videos path/to/video.mp4 --config framework/experiments/auto-annotate.yaml

    # run：目录内全部视频 + smoke 探针
    python -m framework.cleansight_eval.cli.annotate run \
        --videos path/to/videos/ --config ... --max-frames 30

    # convert：标注 JSON + 人工导出 → 训练数据
    python -m framework.cleansight_eval.cli.annotate convert \
        --annotations outputs/annotations \
        --labels-export legacy/yolo-detection/pipeline/raw/exports/project-10-at-2026-07-07-19-32.json \
        --out datasets/cleansight-ActionMixed-auto --split train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ..detection import auto_annotate

REPO_ROOT = Path(__file__).resolve().parents[3]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _expand_videos(raw: str) -> list[Path]:
    """展开 --videos：单文件直接返回，目录返回其中全部视频文件（排序）。"""

    path = Path(raw)
    if path.is_dir():
        videos = sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            raise SystemExit(f"--videos 目录中没有视频文件: {path}")
        return videos
    if path.is_file():
        return [path]
    raise SystemExit(f"--videos 不存在: {path}")


def _cmd_run(args) -> int:
    """run：视频 → legacy 标注 JSON。"""

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"[annotate] 配置文件不存在: {config_path}")
        return 2
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    checkpoints = config.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        print("[annotate] 配置缺少 checkpoints 列表（每项含 path + class_map）")
        return 2
    specs = []
    for spec in checkpoints:
        ckpt_path = Path(spec["path"])
        if not ckpt_path.is_absolute():
            ckpt_path = REPO_ROOT / ckpt_path
        specs.append({"path": ckpt_path, "class_map": spec["class_map"]})

    videos_raw = args.videos or config.get("videos")
    if not videos_raw:
        print("[annotate] 未指定 --videos，且配置缺少 videos（默认视频文件/目录）")
        return 2
    videos_path = Path(videos_raw)
    if not videos_path.is_absolute():
        videos_path = REPO_ROOT / videos_path
    videos = _expand_videos(str(videos_path))
    out_dir = Path(args.out or config.get("out_dir") or "outputs/annotations")
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    runs_dir = Path(args.runs_dir or "outputs/ultralytics_runs")
    if not runs_dir.is_absolute():
        runs_dir = REPO_ROOT / runs_dir

    outputs = auto_annotate.run_auto_annotate(
        videos,
        specs,
        out_dir,
        imgsz=args.imgsz or int(config.get("imgsz", 640)),
        conf=args.conf if args.conf is not None else config.get("conf", 0.25),
        top_k=config.get("top_k") or None,
        max_frames=args.max_frames,
        runs_dir=runs_dir,
        frame_stride=args.frame_stride or int(config.get("frame_stride", 1)),
        track=args.track or bool(config.get("track", False)),
        batch_size=args.batch_size or int(config.get("batch_size", 16)),
        resume=args.resume,
    )
    print(f"[annotate] 完成：{len(outputs)} 个 JSON 写入 {out_dir}")
    return 0


def _cmd_convert(args) -> int:
    """convert：标注 JSON + 人工导出 → 时序训练数据布局。"""

    annotation_dir = Path(args.annotations)
    if not annotation_dir.is_dir():
        print(f"[convert] 标注目录不存在: {annotation_dir}")
        return 2
    labels_export = Path(args.labels_export)
    if not labels_export.is_file():
        print(f"[convert] 人工导出不存在: {labels_export}")
        return 2
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root

    outputs = auto_annotate.convert_annotations(
        annotation_dir, labels_export, out_root, split=args.split
    )
    print(f"[convert] 完成：{len(outputs)} 个视频写入 {out_root}（split={args.split}）")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """构造 run/convert 子命令解析器；参数落在同一个 namespace，handler 接收 args。"""

    p = argparse.ArgumentParser(description="YOLO 自动标注（run：视频→JSON；convert：JSON→训练数据）")
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="视频 → legacy 标注 JSON")
    run_p.add_argument("--videos", default=None, help="视频文件或包含视频的目录（缺省取配置 videos，相对仓库根）")
    run_p.add_argument("--config", required=True, help="自动标注配置 YAML（videos/checkpoints/imgsz/conf/top_k/out_dir）")
    run_p.add_argument("--out", default=None, help="输出目录（默认取配置 out_dir，缺省为 outputs/annotations）")
    run_p.add_argument("--runs-dir", default=None, help="ultralytics 中间产物目录（默认 outputs/ultralytics_runs，Git 忽略）")
    run_p.add_argument("--conf", type=float, default=None, help="全局检测置信度阈值（覆盖配置；配置可写类别级 {类别: 阈值}）")
    run_p.add_argument("--imgsz", type=int, default=None, help="推理输入尺寸（覆盖配置）")
    run_p.add_argument("--max-frames", type=int, default=None, help="每视频最多推理帧数（smoke 探针）")
    run_p.add_argument("--frame-stride", type=int, default=None, help="每 N 帧推理一次，中间帧沿用最近结果（推理成本降 N 倍）")
    run_p.add_argument("--batch-size", type=int, default=None, help="批量推理帧数（默认 16，GPU 利用率更高）")
    run_p.add_argument("--track", action="store_true", help="启用 ByteTrack 实例跟踪（轨迹按实例 id 组织）")
    run_p.add_argument("--resume", action="store_true", help="跳过已存在产出的视频（断点续跑）")
    run_p.set_defaults(handler=_cmd_run)

    convert_p = sub.add_parser("convert", help="标注 JSON + 人工导出 → 时序训练数据")
    convert_p.add_argument("--annotations", required=True, help="自动标注 JSON 目录")
    convert_p.add_argument("--labels-export", required=True, help="人工 Label Studio 导出 JSON（取 timelinelabels 动作标签）")
    convert_p.add_argument("--out", required=True, help="训练数据根目录（labels/ + frames/）")
    convert_p.add_argument("--split", default="train", help="split 名（默认 train；可多次调用生成 train/val）")
    convert_p.set_defaults(handler=_cmd_convert)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
