"""YOLO 自动标注 CLI：python -m framework.cleansight_eval.cli.annotate。

用已训练 YOLO checkpoint 对无标注视频逐帧检测，产出 legacy 时序标注 JSON
（与 Label Studio 导出同构，可被历史 ``lab.py::load_data_json`` 直接消费），
作为时序模型训练的特征输入。

用法（仓库根执行）:
    # 单视频
    python -m framework.cleansight_eval.cli.annotate \
        --videos path/to/video.mp4 --config framework/experiments/auto-annotate.yaml

    # 目录内全部视频
    python -m framework.cleansight_eval.cli.annotate \
        --videos path/to/videos/ --config framework/experiments/auto-annotate.yaml

    # smoke 探针（限制帧数，快速验证链路）
    python -m framework.cleansight_eval.cli.annotate \
        --videos path/to/video.mp4 --config ... --max-frames 30
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="YOLO 自动标注（→ legacy 时序标注 JSON）")
    p.add_argument("--videos", required=True, help="视频文件或包含视频的目录")
    p.add_argument("--config", required=True, help="自动标注配置 YAML（checkpoints/imgsz/conf/top_k/out_dir）")
    p.add_argument("--out", default=None, help="输出目录（默认取配置 out_dir，缺省为 outputs/annotations）")
    p.add_argument("--runs-dir", default=None, help="ultralytics 中间产物目录（默认 outputs/ultralytics_runs，Git 忽略）")
    p.add_argument("--conf", type=float, default=None, help="检测置信度阈值（覆盖配置）")
    p.add_argument("--imgsz", type=int, default=None, help="推理输入尺寸（覆盖配置）")
    p.add_argument("--max-frames", type=int, default=None, help="每视频最多推理帧数（smoke 探针）")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
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

    videos = _expand_videos(args.videos)
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
        conf=args.conf if args.conf is not None else float(config.get("conf", 0.25)),
        top_k=config.get("top_k") or None,
        max_frames=args.max_frames,
        runs_dir=runs_dir,
    )
    print(f"[annotate] 完成：{len(outputs)} 个 JSON 写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
