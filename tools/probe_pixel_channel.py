"""像素通道探针（提案 P1）：冻结 backbone 下"整帧 vs ROI 裁剪"embedding 的四臂线性探针对照。

问题（`docs/features/INPUT_DESIGN_PROPOSAL.md` §2 P1）：E1 的整帧 224² embedding 段级略好但
逐类 recall 崩，怀疑是**尺度取错**（判别物在整帧里只有 9~38 px）。本脚本在机制床
`datasets/cleansight-ActionMixed`（图/框/标签齐全、整帧 embedding 已预计算）上做四臂对照：

  A 整帧 224² embedding（576 维，复用已预计算产物 —— 即 E1 现状）
  B hand 裁剪 embedding（最大 hand 框 ×1.5 扩张，口径同 features/hand_bbox.py）
  C hand + scope 两槽裁剪 embedding（scope = 三个 scope 类的并集框 ×1.2）
  D ROI-144（frames/*.txt 的文本特征，作为"当前主线看不看得见"的参照）

指标：帧级 macro-F1 + 逐类 recall/precision，重点看 insert / withdraw / sb_cleaning；
另附 insert-vs-withdraw 的二分类 AUC（方向判别）。

用法：
    python tools/probe_pixel_channel.py extract    # 抽取 B/C 臂裁剪 embedding（CPU，几分钟）并缓存
    python tools/probe_pixel_channel.py evaluate   # 拟合线性探针并输出四臂对照表
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from framework.cleansight_eval.temporal.features.extract_embeddings import (  # noqa: E402
    _preprocess_tensor,
    build_backbone,
    load_frame_rgb,
)
from framework.cleansight_eval.temporal.features.hand_bbox import (  # noqa: E402
    HAND_CLASS_ID,
    HAND_REGION_EXPAND,
)
from framework.cleansight_eval.temporal.features.roi_bbox import (  # noqa: E402
    build_roi_frame_features,
)

BED = ROOT / "datasets/cleansight-ActionMixed"
WORK = ROOT / "tmp/pixel_probe"
SCOPE_CLASS_IDS = (1, 2, 3)   # scope_control_body / scope_mid_section / scope_distal_end
SCOPE_EXPAND = 1.2
SPLITS = ("train", "val", "test")
FOCUS = ("long_brush_insert", "long_brush_withdraw", "short_brush_cleaning")
ACTION_NAMES = ["idle", "air_injection", "flush", "long_brush_insert",
                "long_brush_withdraw", "short_brush_cleaning"]
EMBED_ROOT = BED / "embeddings/mobilenet_v3_small-v1"


# ---------------------------------------------------------------- 数据读取

def read_frames(split: str):
    """按视频读标签序列与检测框，返回 [(stem, frame_ids, action_ids, boxes_per_frame)]。"""

    out = []
    for label_path in sorted((BED / "labels" / split).glob("*.txt")):
        stem = label_path.name[:-4]                    # 去掉 .txt → xxx.mp4
        frames, actions = [], []
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                frames.append(int(parts[0]))
                actions.append(int(parts[1]))
        boxes = []
        for frame_id in frames:
            txt = BED / "frames" / split / f"{stem}-{frame_id:06d}.txt"
            parsed = []
            if txt.is_file():
                for line in txt.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 5:
                        parsed.append(tuple(float(v) for v in parts))
            boxes.append(parsed)
        out.append((stem, frames, np.asarray(actions), boxes))
    return out


def image_path(split: str, stem: str, frame_id: int) -> pathlib.Path:
    return BED / "images" / split / f"{stem}-{frame_id:06d}.jpg"


# ---------------------------------------------------------------- 区域规则

def hand_region(boxes) -> tuple[float, float, float, float] | None:
    """最大 hand 框 ×HAND_REGION_EXPAND 扩张并钳制（口径同 features/hand_bbox.py）。"""

    best, best_area = None, -1.0
    for box in boxes:
        cls, cx, cy, w, h = box
        if int(cls) != HAND_CLASS_ID:
            continue
        area = w * h
        if area > best_area:
            best, best_area = box, area
    if best is None:
        return None
    _cls, cx, cy, w, h = best
    hw, hh = w / 2.0, h / 2.0
    return (max(0.0, cx - hw * HAND_REGION_EXPAND), max(0.0, cy - hh * HAND_REGION_EXPAND),
            min(1.0, cx + hw * HAND_REGION_EXPAND), min(1.0, cy + hh * HAND_REGION_EXPAND))


def scope_region(boxes) -> tuple[float, float, float, float] | None:
    """三个 scope 类检测框的并集框 ×SCOPE_EXPAND 并钳制；无该类框返回 None。"""

    xs1, ys1, xs2, ys2 = [], [], [], []
    for box in boxes:
        cls, cx, cy, w, h = box
        if int(cls) not in SCOPE_CLASS_IDS:
            continue
        xs1.append(cx - w / 2.0)
        ys1.append(cy - h / 2.0)
        xs2.append(cx + w / 2.0)
        ys2.append(cy + h / 2.0)
    if not xs1:
        return None
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) / 2.0 * SCOPE_EXPAND, (y2 - y1) / 2.0 * SCOPE_EXPAND
    return (max(0.0, cx - hw), max(0.0, cy - hh), min(1.0, cx + hw), min(1.0, cy + hh))


def load_crop_rgb(path: pathlib.Path, box, size: int = 224):
    """按归一化框裁剪并 resize；文件缺失或框非法返回 None。"""

    import cv2

    if box is None or not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        return None
    height, width = image.shape[:2]
    x1 = int(round(box[0] * width))
    y1 = int(round(box[1] * height))
    x2 = int(round(box[2] * width))
    y2 = int(round(box[3] * height))
    x1, y1 = max(0, min(x1, width - 2)), max(0, min(y1, height - 2))
    x2, y2 = max(x1 + 2, min(x2, width)), max(y1 + 2, min(y2, height))
    crop = image[y1:y2, x1:x2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- 抽取

def extract(device: str = "cpu") -> None:
    import torch

    model, size, feat_dim = build_backbone("mobilenet_v3_small", device)
    print(f"backbone=mobilenet_v3_small size={size} feat_dim={feat_dim}", flush=True)
    for split in SPLITS:
        for stem, frame_ids, actions, boxes in read_frames(split):
            out_dir = WORK / "emb" / split
            out_dir.mkdir(parents=True, exist_ok=True)
            if (out_dir / f"{stem}.hand.npy").is_file():
                continue
            hand_feats, scope_feats, flags = [], [], []
            for frame_id, frame_boxes in zip(frame_ids, boxes):
                path = image_path(split, stem, frame_id)
                hand = hand_region(frame_boxes)
                scope = scope_region(frame_boxes)
                flags.append([1.0 if hand else 0.0, 1.0 if scope else 0.0])
                hand_feats.append(load_crop_rgb(path, hand, size))
                scope_feats.append(load_crop_rgb(path, scope, size))

            def embed(crops):
                """逐帧裁剪图 → [T, feat_dim]；缺失帧补零。"""

                out = np.zeros((len(crops), feat_dim), dtype=np.float32)
                index = [i for i, c in enumerate(crops) if c is not None]
                for start in range(0, len(index), 64):
                    batch = index[start:start + 64]
                    tensor = _preprocess_tensor([crops[i] for i in batch], size, device)
                    with torch.no_grad():
                        out[batch] = model(tensor).cpu().numpy()
                return out

            np.save(out_dir / f"{stem}.hand.npy", embed(hand_feats))
            np.save(out_dir / f"{stem}.scope.npy", embed(scope_feats))
            np.save(out_dir / f"{stem}.flags.npy", np.asarray(flags, dtype=np.float32))
            print(f"  {split}/{stem}: {len(frame_ids)} 帧", flush=True)


# ---------------------------------------------------------------- 评估

def fit_lda(x: np.ndarray, y: np.ndarray, n_classes: int, ridge: float = 1e-2):
    """等先验多类 LDA（闭式解，带 ridge 稳定项）→ (w, b)。"""

    dim = x.shape[1]
    mus, covs, counts = [], [], []
    for c in range(n_classes):
        xc = x[y == c]
        mus.append(xc.mean(axis=0) if len(xc) else np.zeros(dim))
        covs.append(np.cov(xc, rowvar=False) if len(xc) > 1 else np.zeros((dim, dim)))
        counts.append(len(xc))
    mu = np.stack(mus)
    pooled = sum((n - 1) * cov for n, cov in zip(counts, covs)) / max(sum(counts) - n_classes, 1)
    inv = np.linalg.inv(pooled + ridge * (float(np.trace(pooled)) / dim or 1.0) * np.eye(dim))
    return (inv @ mu.T).T, -0.5 * np.einsum("cd,cd->c", mu @ inv, mu)


def metrics(pred: np.ndarray, truth: np.ndarray, n_classes: int, names: list[str]) -> dict:
    """帧级 macro-F1 + 逐类 recall/precision。"""

    per_class, f1s = {}, []
    for c in range(n_classes):
        tp = int(((truth == c) & (pred == c)).sum())
        support = int((truth == c).sum())
        predicted = int((pred == c).sum())
        recall = None if support == 0 else tp / support
        precision = None if predicted == 0 else tp / predicted
        f1 = None if not recall or not precision else 2 * recall * precision / (recall + precision)
        per_class[names[c]] = {"support": support, "predicted": predicted,
                               "recall": recall, "precision": precision}
        if f1 is not None:
            f1s.append(f1)
    return {"macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
            "acc": float((pred == truth).mean()), "per_class": per_class}


def load_arm(name: str, split: str, frames_cache: dict):
    """返回该臂在该 split 上的 (逐帧特征, 逐帧真值, 视频名)。"""

    rows, truths, videos = [], [], []
    for stem, frame_ids, actions, boxes in frames_cache[split]:
        if name == "A_whole":
            emb = np.load(EMBED_ROOT / split / f"{stem}.npy").astype(np.float64)
            rows.append(emb)
        elif name == "B_hand":
            emb = np.load(WORK / "emb" / split / f"{stem}.hand.npy").astype(np.float64)
            flags = np.load(WORK / "emb" / split / f"{stem}.flags.npy")[:, :1]
            rows.append(np.concatenate([emb, flags], axis=1))
        elif name == "C_hand_scope":
            hand = np.load(WORK / "emb" / split / f"{stem}.hand.npy").astype(np.float64)
            scope = np.load(WORK / "emb" / split / f"{stem}.scope.npy").astype(np.float64)
            flags = np.load(WORK / "emb" / split / f"{stem}.flags.npy")
            rows.append(np.concatenate([hand, scope, flags], axis=1))
        elif name == "D_roi144":
            feats = [build_roi_frame_features(BED / "frames" / split / f"{stem}-{fid:06d}.txt")
                     for fid in frame_ids]
            rows.append(np.stack(feats).astype(np.float64))
        else:
            raise SystemExit(f"未知臂 {name}")
        truths.append(actions)
        videos.append(stem)
    return np.concatenate(rows), np.concatenate(truths), videos


def evaluate() -> None:
    frames_cache = {split: read_frames(split) for split in SPLITS}
    arms = ["A_whole", "B_hand", "C_hand_scope", "D_roi144"]
    report = {}
    lines = ["# P1 探针：整帧 vs ROI 裁剪（机制床 cleansight-ActionMixed）", "",
             "冻结 mobilenet_v3_small（ImageNet 预训练）+ 等先验多类 LDA（train 拟合，val/test 评估）。",
             "A=整帧 224²、B=hand 裁剪(+presence)、C=hand+scope 两槽(+presence)、D=ROI-144 文本特征。", "",
             "| 臂 | 维度 | split | macro-F1 | acc | " + " | ".join(ACTION_NAMES) + " |",
             "|---|---:|---|---:|---:|" + "---:|" * len(ACTION_NAMES)]
    for arm in arms:
        x_tr, y_tr, _ = load_arm(arm, "train", frames_cache)
        w, b = fit_lda(x_tr, y_tr, len(ACTION_NAMES))
        report[arm] = {"dim": int(x_tr.shape[1])}
        for split in ("val", "test"):
            x, y, videos = load_arm(arm, split, frames_cache)
            pred = np.argmax(x @ w.T + b, axis=1)
            m = metrics(pred, y, len(ACTION_NAMES), ACTION_NAMES)
            report[arm][split] = m
            cells = []
            for name in ACTION_NAMES:
                entry = m["per_class"][name]
                if entry["support"] == 0:
                    cells.append("n/a")
                else:
                    rec = f"{entry['recall'] * 100:.0f}"
                    prec = "—" if entry["precision"] is None else f"{entry['precision'] * 100:.0f}"
                    cells.append(f"{rec}/{prec}")
            lines.append(f"| {arm} | {x_tr.shape[1]} | {split} | {m['macro_f1'] * 100:.1f} | "
                         f"{m['acc'] * 100:.1f} | " + " | ".join(cells) + " |")
            print(f"{arm:14} {split:5} macroF1={m['macro_f1'] * 100:5.1f} acc={m['acc'] * 100:5.1f} "
                  + " ".join(f"{n[:6]}={'' if m['per_class'][n]['recall'] is None else format(m['per_class'][n]['recall'] * 100, '.0f')}"
                             for n in ACTION_NAMES), flush=True)
        lines.append("")
    lines += ["（逐类单元格为 `recall/precision`；support=0 的类记 n/a。",
              "重点看 long_brush_insert / long_brush_withdraw / short_brush_cleaning 三列。）"]
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "P1_PROBE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (WORK / "p1_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {WORK / 'P1_PROBE.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="像素通道探针：整帧 vs ROI 裁剪 embedding")
    parser.add_argument("phase", choices=["extract", "evaluate"], help="extract=抽裁剪 embedding，evaluate=出对照表")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.phase == "extract":
        extract(args.device)
    else:
        evaluate()


if __name__ == "__main__":
    main()
