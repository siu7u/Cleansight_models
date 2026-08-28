"""时序模型预测直达工具：一个 .pt → 动作段时间线 + 带状图 + 叠加预测视频。

用途：训练好的时序 checkpoint（如 ``registry/temporal/auto-mstcn-v1`` 指向的
``runs/*/checkpoints/best.pt``）对数据集里一个/全部序列直接预测，产出三样东西
用于人工判断"结果是否可用"：

1. **动作段时间线**（stdout + ``<out-dir>/timeline_<序列>.json``）：预测按连续
   同类帧合并成段，给出每段的动作名、起止真实帧号与帧数；
2. **GT/Pred 带状图**（``<out-dir>/segmentation-<split>-pNN.png``）：复用
   ``benchmark.visualizers.temporal``，逐视频对照人工真值与预测；
3. **叠加预测视频**（``<out-dir>/<序列>_pred.mp4``）：顶部横幅显示当前帧预测
   动作阶段，复用 ``tools/visualize_predictions`` 的渲染（帧图来自 --images 或
   --video）。

模型配置（类型/窗口/特征映射）从 checkpoint 旁的 ``<ckpt>.meta.json`` sidecar
自动读取，不需要手写 YAML；推理走 framework 时序流水线的公开 ``predict()``。

用法：
    # 对 auto-mstcn-v1 权重预测整个 train split（不指定 --sequence）
    python tools/predict_timeline.py \
        --ckpt runs/mstcn-20260817-170254/checkpoints/best.pt \
        --dataset datasets/cleansight-ActionMixed-auto --split train \
        --images datasets/cleansight-ActionMixed/images/train \
        --out-dir outputs/visualizations/pred_auto_mstcn

    # 只预测一个序列（--split 缺省自动探测；像素源也可用 --video）
    python tools/predict_timeline.py --ckpt ... --dataset ... \
        --sequence 05ba4406-clip_....mp4 \
        --images datasets/cleansight-ActionMixed/images/train --out-dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 脚本直跑（python tools/predict_timeline.py）时把仓库根加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from framework.cleansight_eval.core.execution import PredictionOutput

try:
    from tools.visualize_predictions import render_artifact_videos
except ImportError:  # 以脚本方式运行（python tools/xxx.py）时 tools/ 在 sys.path
    from visualize_predictions import render_artifact_videos

IDLE_NAME = "idle"


def load_meta(ckpt: Path) -> dict:
    """读 ``<ckpt>.meta.json`` sidecar（模型类型/窗口/特征映射等重建信息）。"""

    meta_path = ckpt.with_suffix(ckpt.suffix + ".meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"checkpoint 缺少 meta sidecar: {meta_path}（本工具只支持带 meta 的时序权重）"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def build_cfg(meta: dict, dataset_root: Path, split: str) -> dict:
    """由 meta sidecar + 数据集根构造最小评估配置（data/model/feature_schema）。"""

    cfg = {
        "model": meta["model"],
        "data": {
            "root": str(dataset_root),
            "split_eval": split,
            "labels_dir": "labels",
            "frames_dir": "frames",
        },
        "feature_schema": meta.get("feature_schema"),
        "evaluation": {"mode": "formal", "measure_latency": False},
        "train": {"window": meta.get("window") or 64},
    }
    return cfg


def _pipeline_for(meta: dict):
    """按 ``meta["pipeline"]`` 选择时序流水线实现（滑窗/全序列）。"""

    from framework.cleansight_eval.temporal.full_sequence_pipeline import (
        FullSequenceTemporalPipeline,
    )
    from framework.cleansight_eval.temporal.sliding_window_pipeline import (
        SlidingWindowTemporalPipeline,
    )

    if meta.get("pipeline") == "sliding_window_temporal":
        return SlidingWindowTemporalPipeline()
    return FullSequenceTemporalPipeline()


def run_predict(meta: dict, cfg: dict, ckpt: Path, device: str) -> PredictionOutput:
    """执行一次流水线预测，返回逐视频预测/真值事实（不含指标判分）。"""

    return _pipeline_for(meta).predict(cfg, str(ckpt), device)


def derive_segments(predicted_ids, frame_ids: list[int], id2name: dict) -> list[dict]:
    """把逐抽样帧预测合并成连续动作段，返回 ``[{action, action_id, start_frame, end_frame, num_frames}]``。

    ``predicted_ids`` 为按抽样帧序的类别 id（与 ``frame_ids`` 一一对应）；
    ``id2name`` 为 ``{id: 动作名}``。相邻同类别抽样帧合并为一段；段边界用
    抽样帧的真实帧号（起止）标注。
    """

    segments: list[dict] = []
    for index, pred_id in enumerate(predicted_ids):
        pred_id = int(pred_id)
        if segments and segments[-1]["action_id"] == pred_id:
            segments[-1]["end_frame"] = frame_ids[index]
            segments[-1]["num_frames"] += 1
        else:
            segments.append(
                {
                    "action": id2name.get(pred_id, f"cls_{pred_id}"),
                    "action_id": pred_id,
                    "start_frame": frame_ids[index],
                    "end_frame": frame_ids[index],
                    "num_frames": 1,
                }
            )
    return segments


def build_prediction_artifact(pred_output: PredictionOutput, names: list[str]) -> dict:
    """把 PredictionOutput 的指定序列转成 prediction-artifact-v1 结构的 dict。

    产出直接喂给 ``tools.visualize_predictions.render_artifact_videos`` 渲染视频；
    ``predicted_label_ids``/``truth_label_ids`` 由标签名按 ``labels`` 顺序回编。
    """

    labels = list(pred_output.labels)
    name_to_id = {str(name): index for index, name in enumerate(labels)}
    items: dict = {}
    for name in names:
        predicted = list(pred_output.predictions[name])
        targets = list(pred_output.targets.get(name, []))
        items[name] = {
            "prediction_start_frame": 0,
            "num_predictions": len(predicted),
            "predicted_label_ids": [name_to_id[str(value)] for value in predicted],
            "truth_label_ids": [name_to_id[str(value)] for value in targets],
            "predicted_labels": [str(value) for value in predicted],
            "truth_labels": [str(value) for value in targets],
        }
    return {
        "schema_version": 1,
        "task_type": "temporal",
        "prediction_format": "frame_labels",
        "inference": {"mode": str(getattr(pred_output, "pipeline", "temporal"))},
        "labels": [{"id": index, "name": name} for index, name in enumerate(labels)],
        "items": items,
    }


def _resolve_split(dataset_root: Path, sequence: str | None, split: str | None) -> str:
    """确定 split：显式 --split 优先；只给序列时在 labels/ 下自动探测唯一命中。"""

    if split:
        return split
    if sequence is None:
        return "train" if (dataset_root / "labels" / "train").is_dir() else "test"
    labels_root = dataset_root / "labels"
    hits = sorted(
        d.name for d in labels_root.iterdir()
        if d.is_dir() and (d / f"{sequence}.txt").is_file()
    ) if labels_root.is_dir() else []
    if len(hits) == 1:
        return hits[0]
    if "test" in hits:
        return "test"
    if hits:
        return hits[0]
    raise FileNotFoundError(f"数据集 labels/ 下找不到序列 {sequence} 的动作标签")


def _write_segment_timeline(out_dir: Path, sequence: str, segments: list[dict], frame_acc: float) -> Path:
    """写 ``<out-dir>/timeline_<序列>.json`` 并打印人类可读时间线，返回 JSON 路径。"""

    payload = {"sequence": sequence, "frame_acc": frame_acc, "segments": segments}
    path = out_dir / f"timeline_{Path(sequence).stem}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[predict-timeline] === {sequence} 动作段时间线"
        f"（frame-acc={frame_acc * 100:.1f}%）==="
    )
    for index, segment in enumerate(segments, start=1):
        print(
            f"[predict-timeline]   {index:>2}. {segment['action']:<22} "
            f"帧 {segment['start_frame']}-{segment['end_frame']} "
            f"（{segment['num_frames']} 帧）"
        )
    return path


def run_predict_timeline(
    ckpt: Path,
    dataset_root: Path,
    *,
    sequence: str | None = None,
    split: str | None = None,
    images_dir: Path | None = None,
    video_path: Path | None = None,
    out_dir: Path,
    device: str | None = None,
    max_frames: int = 0,
    fps: float = 0.0,
    draw_boxes: bool = True,
) -> dict:
    """主入口：.pt + 数据集 → 时间线 JSON + 带状图 + 预测视频，返回汇总信息。"""

    import torch

    from benchmark.visualizers.temporal import render_prediction_timeline

    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(ckpt)
    split = _resolve_split(dataset_root, sequence, split)
    cfg = build_cfg(meta, dataset_root, split)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    pred_output = run_predict(meta, cfg, ckpt, device)
    available = sorted(pred_output.predictions)
    names = [sequence] if sequence else available
    missing = [name for name in names if name not in pred_output.predictions]
    if missing:
        raise KeyError(f"预测结果中没有序列 {missing[0]}；可用: {', '.join(available[:10])}")

    id2name = {index: str(name) for index, name in enumerate(pred_output.labels)}
    name_to_id = {str(name): index for index, name in enumerate(pred_output.labels)}
    frame_accs: dict[str, float] = {}
    timeline_paths: list[Path] = []
    for name in names:
        label_path = dataset_root / "labels" / split / f"{name}.txt"
        frame_ids = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    frame_ids.append(int(parts[0]))
                except ValueError:
                    continue
        predicted_ids = [name_to_id[str(value)] for value in pred_output.predictions[name]]
        truth_ids = [name_to_id[str(value)] for value in pred_output.targets.get(name, [])]
        if len(predicted_ids) != len(frame_ids):
            raise ValueError(
                f"{name}: 预测长度 {len(predicted_ids)} 与标签帧数 {len(frame_ids)} 不一致"
            )
        segments = derive_segments(predicted_ids, frame_ids, id2name)
        frame_acc = float(
            np.mean([p == t for p, t in zip(predicted_ids, truth_ids)]) if truth_ids else 0.0
        )
        frame_accs[name] = frame_acc
        timeline_paths.append(_write_segment_timeline(out_dir, name, segments, frame_acc))

    # 带状图（GT vs Pred，只画选中的序列）
    viz = render_prediction_timeline(
        {
            "predictions": {name: pred_output.predictions[name] for name in names},
            "targets": {name: pred_output.targets.get(name, []) for name in names},
            "labels": list(pred_output.labels),
            "metadata": {"split": split},
            "model_type": meta.get("type", "temporal"),
        },
        out_dir=out_dir,
    )

    # 叠加预测视频（复用 visualize_predictions 的渲染；数据集 labels/ 缺失的序列会跳过）
    artifact = build_prediction_artifact(pred_output, names)
    videos = render_artifact_videos(
        artifact,
        dataset_root,
        images_dir=images_dir,
        video_path=video_path,
        out_dir=out_dir,
        sequence=None,
        split=split,
        max_frames=max_frames,
        fps=fps,
        draw_boxes=draw_boxes,
    )

    return {
        "ckpt": str(ckpt),
        "pipeline": meta.get("pipeline"),
        "split": split,
        "device": device,
        "frame_acc": frame_accs,
        "timelines": [str(path) for path in timeline_paths],
        "band_charts": [str(path) for path in viz],
        "videos": [str(path) for path in videos],
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="时序 .pt + 数据集 → 动作段时间线 + 带状图 + 叠加预测视频（不判分）"
    )
    p.add_argument("--ckpt", required=True, help="时序 checkpoint（需带 <ckpt>.meta.json sidecar）")
    p.add_argument("--dataset", required=True, help="数据集根（labels/<split>/ + frames/<split>/ 布局）")
    p.add_argument("--sequence", default=None, help="只预测指定序列（缺省预测 split 内全部序列）")
    p.add_argument("--split", default=None, help="数据集 split（缺省：--sequence 时自动探测，否则 train）")
    p.add_argument("--images", default=None, help="图片帧序列目录（<序列>-<帧号:06d>.jpg）")
    p.add_argument("--video", default=None, help="原视频路径（按真实帧号抽取帧图）")
    p.add_argument("--out-dir", default="outputs/visualizations", help="输出目录（时间线/带状图/视频）")
    p.add_argument("--device", default=None, help="推理设备（缺省自动：cuda 可用则用，否则 cpu）")
    p.add_argument("--max-frames", type=int, default=0, help="每视频最多渲染的标签帧数（0=全部）")
    p.add_argument("--fps", type=float, default=0.0, help="视频输出帧率（0=自动）")
    p.add_argument("--no-boxes", action="store_true", help="视频不叠加 YOLO 检测框")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = run_predict_timeline(
        Path(args.ckpt),
        Path(args.dataset),
        sequence=args.sequence,
        split=args.split,
        images_dir=Path(args.images) if args.images else None,
        video_path=Path(args.video) if args.video else None,
        out_dir=Path(args.out_dir),
        device=args.device,
        max_frames=args.max_frames,
        fps=args.fps,
        draw_boxes=not args.no_boxes,
    )
    print(
        f"[predict-timeline] 完成: {len(result['videos'])} 个视频 -> {result['videos']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
