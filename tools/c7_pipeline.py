# -*- coding: utf-8 -*-
"""C7 生产推理管线：视频 → YOLO 框 → 手部角色 → MS-TCN2 → v1.2 后处理 → 动作段。

组合依据（docs/C7_ROLE_HANDS_EVALUATION_20261017.md）：
  C7 概率 + 我方 v1.2 后处理 = test edit 71.8 / F1@0.25 69.3（离线最优）。

用法：
  # 已有检测框（auto-annotate 产物）
  python tools/c7_pipeline.py --video V.mp4 --frames-dir frames/test \
      --stem V.mp4 --out seg.json

  # 复用手部嵌入缓存（首次会算 DINOv2，CPU 约 2-4 分钟/视频）
  python tools/c7_pipeline.py ... --embeddings-cache tmp/c7_hand_embeddings

依赖：
  - C7 交付包目录（默认 role-hands-c7-mstcn2-v41/，权重 84MB 不入库单独管理）
  - venv: numpy/torch/opencv；DINOv2 首次联网下载（已缓存则离线）

已知限制（评估文档 §七-九）：
  - 仅离线（角色平滑用未来 0.5s + 双向卷积 + 中心窗平滑）
  - 新视频用全部 20 权重平均的效果未经单独评测（正式上线建议全量重训）
  - 纯 flush 视频可能幻觉长毛刷动作（67edf4f9 案例）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "role-hands-c7-mstcn2-v41"))

from c7ref.features import build_features, hand_boxes_per_frame  # noqa: E402
from c7ref.frames import frame_paths  # noqa: E402
from c7ref.hand_state import HandStateClassifier, embed_hands  # noqa: E402
from c7ref.model import load_model, predict_probs  # noqa: E402
from framework.cleansight_eval.temporal.postprocess import (  # noqa: E402
    DEFAULT_MAX_LEN,
    DEFAULT_MIN_LEN,
    DEFAULT_SMOOTH_K,
    apply_v12,
)

ACTION_NAMES = [
    "idle", "water_injection", "flush",
    "long_brush_insert", "long_brush_withdraw", "short_brush_cleaning",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", required=True, help="视频文件路径")
    ap.add_argument("--frames-dir", required=True,
                    help="检测框目录（YOLO auto-annotate 产物，txt 每行 class cx cy w h）")
    ap.add_argument("--stem", required=True, help="视频名（含 .mp4，与框文件前缀一致）")
    ap.add_argument("--c7-dir", default="role-hands-c7-mstcn2-v41", help="C7 交付包目录")
    ap.add_argument("--weights", default="all",
                    help="all=20 权重平均（新视频默认）| fold<k>=该折 4 seed（复现评测用）")
    ap.add_argument("--hand-state", default="all",
                    help="all=全量分类器（新视频默认）| fold<k>=折排除版（复现评测用）")
    ap.add_argument("--embeddings-cache", default="tmp/c7_hand_embeddings",
                    help="手部嵌入缓存目录（避免重复算 DINOv2）")
    ap.add_argument("--postprocess", choices=["v12", "raw"], default="v12",
                    help="v1.2 后处理（默认，逐类 min/max + 平滑）或 raw")
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    ap.add_argument("--save-probs", action="store_true", help="输出附加逐帧概率")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--labels", default=None,
                    help="标签文件（帧号 类id）；提供时输出附带与 GT 的对照指标")
    return ap.parse_args()


def collapse_segments(labels: np.ndarray) -> list[list]:
    segs, start = [], 0
    values = labels.tolist()
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            segs.append([start, i, ACTION_NAMES[values[start]]])
            start = i
    return segs


def main() -> int:
    args = parse_args()
    c7_dir = ROOT / args.c7_dir
    if not c7_dir.exists():
        print(f"ERROR: C7 交付包不存在: {c7_dir}", file=sys.stderr)
        return 1

    # 帧号：标签文件优先（README 约定防时间轴压缩），否则按框文件枚举
    label_file = Path(args.labels) if args.labels else None
    if label_file and label_file.exists():
        lines = [l.split() for l in label_file.read_text().splitlines() if l.strip()]
        fids = [int(p[0]) for p in lines]
    else:
        from c7ref.frames import frame_ids_from_dir
        fids = frame_ids_from_dir(Path(args.frames_dir), args.stem)
    if not fids:
        print("ERROR: 无可用帧号（检查 frames-dir/stem）", file=sys.stderr)
        return 1
    paths = frame_paths(Path(args.frames_dir), args.stem, fids)

    # ① 手部嵌入（缓存优先）
    cache = ROOT / args.embeddings_cache
    cache.mkdir(parents=True, exist_ok=True)
    emb_path = cache / f"{Path(args.stem).stem[:8]}.npy"
    if emb_path.exists():
        emb = np.load(emb_path)
    else:
        print(f"[c7] 计算 DINOv2 手部嵌入 ({args.device})...", file=sys.stderr)
        emb = embed_hands(args.video, fids, hand_boxes_per_frame(paths), device=args.device)
        np.save(emb_path, emb.astype(np.float16))
        emb = emb.astype(np.float16)

    # ② 手部状态分类器
    hs_name = "all.npz" if args.hand_state == "all" else f"fold{args.hand_state[4:]}.npz"
    hs_path = c7_dir / "weights/hand_state" / hs_name
    hand_probs = HandStateClassifier(str(hs_path)).probs(emb.astype("float64"))

    # ③ 106 维特征 + 模型
    feats = build_features(paths, hand_probs)
    if args.weights == "all":
        weight_files = sorted((c7_dir / "weights/temporal").glob("fold*-seed*.pt"))
    else:
        weight_files = sorted((c7_dir / "weights/temporal").glob(f"{args.weights}-seed*.pt"))
    if not weight_files:
        print(f"ERROR: 无匹配权重 ({args.weights})", file=sys.stderr)
        return 1
    probs = predict_probs([load_model(str(w)) for w in weight_files], feats)

    # ④ 后处理 + 段
    if args.postprocess == "v12":
        labels = apply_v12(probs)
        decode_config = {
            "decoder": "v1.2",
            "min_len": list(DEFAULT_MIN_LEN),
            "max_len": list(DEFAULT_MAX_LEN),
            "smooth_k": DEFAULT_SMOOTH_K,
            "merge_mode": "idle_preferred",
        }
    else:
        labels = probs.argmax(axis=1)
        decode_config = {"decoder": "raw"}

    segments = collapse_segments(labels)
    result = {
        "video": args.stem,
        "n_frames": len(fids),
        "frame_ids": {"first": fids[0], "last": fids[-1], "stride": "15fps(30fps 每2取1)"},
        "segments": segments,
        "segments_raw": collapse_segments(probs.argmax(axis=1)),
        "decode_config": decode_config,
        "model": {
            "type": "c7-mstcn2-rolehands",
            "weights": f"{len(weight_files)} files ({args.weights})",
            "hand_state": hs_name,
            "feature_dim": int(feats.shape[1]),
        },
    }
    if args.save_probs:
        result["probs"] = probs.round(6).tolist()

    # 可选：GT 对照
    if label_file and label_file.exists():
        from framework.cleansight_eval.core.metrics import temporal_metrics
        gt = np.array([int(p[1]) for p in lines], dtype=np.int64)
        r = temporal_metrics(
            {"v": [ACTION_NAMES[x] for x in labels]},
            {"v": [ACTION_NAMES[x] for x in gt]},
            ACTION_NAMES,
        )
        seg = r["segment"]
        f25 = seg.get("details_at_iou", {}).get("0.25", {}).get("f1")
        result["metrics_vs_gt"] = {
            "edit": round(float(seg["edit"]) * 100, 2),
            "f1@0.25": round(float(f25) * 100, 2) if f25 else None,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    n_action = sum(1 for s in segments if s[2] != "idle")
    print(f"[c7] {args.stem}: {len(segments)} 段（动作 {n_action}）→ {out}")
    if "metrics_vs_gt" in result:
        print(f"[c7] 对照: {result['metrics_vs_gt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
