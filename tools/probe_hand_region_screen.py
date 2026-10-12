"""手部区域 recipe 的**源域筛选**（只碰 train，不碰 test）：锚点规则 × 扩张倍数 × 是否带全画面段。

**为什么要筛这条轴**：v4.1 全量重训的 acc 排名给出一个很硬的信号——**手部区域特征是最强的
帧准确率驱动**：`actionmixed-bbox-8cls-v1`（全画面 40 维）acc **69.38**，加进手部区域段后
`...-global-hand-8cls-v1`（80 维）acc **76.27**（+6.89pp）；只留手部段的 `...-hand-8cls-v1`
也有 **75.98**。而这个 recipe 的两个关键选择**从来没有被调过**：

1. `HAND_REGION_EXPAND = 1.5`（把最大 hand 框绕中心扩张的倍数）——写死在
   `features/hand_bbox.py` 里的常数，v1 之后没动过；
2. 锚点 = **面积最大的单个 hand 框**——多手/漏检时区域会偏。

本脚本在**源域**（auto **train** 28 视频、视频级留出、HistGB）上把这两条一起筛，
报"留出视频的逐帧 acc − 该留出集的全 idle 基线"，并与**现行 recipe**（1.5 × 最大框）
做逐折配对。筛选不碰 test，胜者再用 mstcn2 训练验证。

用法：
    python tools/probe_hand_region_screen.py                      # 全表（10 折 × 留出 6 视频）
    python tools/probe_hand_region_screen.py --folds 5 --held 8   # 更快的粗筛
    python tools/probe_hand_region_screen.py --root datasets/cleansight-ActionMixed-auto-lhh

判读：Δ = **留出视频的逐帧 acc − 该折全 idle 基线**（HistGB 在其余视频上训练），
逐折配对比较用 Wilcoxon；结论只有"源域方向"，最终必须用 mstcn2 训练在 test 上验证
（本仓库历史上有源域胜者不迁移的例子）。2026-10-11 用本工具筛出"并集锚点"并落地为
`actionmixed-bbox-hand-8cls-v2` / `-global-hand-8cls-v2` 两个契约。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]  # tools/ → 仓库根
DEFAULT_ROOT = REPO / "datasets/cleansight-ActionMixed-auto-lhh"
AUTO = DEFAULT_ROOT  # 由 main() 的 --root 覆盖
N_CLASSES = 8
HAND_CLASS_ID = 0
TOOL_CLASS_IDS = (3, 4, 5, 6, 7)   # scope_distal_end / syringe / air_gun / short_brush / brush_tip_out
BRUSH_CLASS_IDS = (6, 7)           # short_brush / brush_tip_out（`short_brush_cleaning` 的器械）
ACTION_NAMES = ("idle", "water_injection", "flush", "long_brush_insert",
                "long_brush_withdraw", "short_brush_cleaning")
EXPANDS = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
# 区域锚点：最大 hand 框 / 全部 hand 框并集 / 并集再并入全部器械框 / 并集再并入刷类框。
# 后两者用来回答"把器械（尤其 short_brush）圈进区域，能否补上只靠手部区域拿不到的稀类证据"。
ANCHORS = ("largest", "union", "union_tools", "union_brush")


def read_train(root: Path | None = None) -> dict[str, tuple[list, np.ndarray]]:
    """→ {stem: (逐帧框列表, 逐帧动作标签)}；框 = [(类, cx, cy, w, h)]。"""

    out = {}
    base = Path(root) if root is not None else AUTO
    for label_file in sorted((base / "labels/train").glob("*.txt")):
        stem = label_file.name[:-4]
        rows = []
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                rows.append((int(parts[0]), int(parts[1])))
        frames = []
        for frame_id, _action in rows:
            path = base / "frames/train" / f"{stem}-{frame_id:06d}.txt"
            boxes = []
            if path.exists():
                for line in path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 5:
                        boxes.append((int(float(parts[0])), *(float(v) for v in parts[1:])))
            frames.append(boxes)
        out[stem] = (frames, np.array([a for _, a in rows], dtype=np.int64))
    return out


def global_block(boxes) -> np.ndarray:
    """全画面 40 维（= `featurize_frame_bbox`）：每类面积最大框 [presence, cx, cy, w, h]。"""

    feat = np.zeros((N_CLASSES, 5), dtype=np.float32)
    best = np.zeros(N_CLASSES, dtype=np.float32)
    for c, cx, cy, w, h in boxes:
        if not (0 <= c < N_CLASSES):
            continue
        area = w * h
        if area >= best[c]:
            best[c] = area
            feat[c] = (1.0, cx, cy, w, h)
    return feat.reshape(-1)


def _expand_union(rects, expand: float):
    """若干 ``(cx, cy, w, h)`` 的并集绕其中心扩张 ``expand`` 倍 → ``(x1, y1, x2, y2)``（钳制画面）。

    只有一个矩形时与"该框绕中心扩张"逐位等价；空列表返回 ``None``。
    """

    if not rects:
        return None
    x1r = min(cx - w / 2 for cx, _cy, w, _h in rects)
    x2r = max(cx + w / 2 for cx, _cy, w, _h in rects)
    y1r = min(cy - h / 2 for _cx, cy, _w, h in rects)
    y2r = max(cy + h / 2 for _cx, cy, _w, h in rects)
    center_x, center_y = (x1r + x2r) / 2, (y1r + y2r) / 2
    half_w, half_h = (x2r - x1r) / 2 * expand, (y2r - y1r) / 2 * expand
    return (max(0.0, center_x - half_w), max(0.0, center_y - half_h),
            min(1.0, center_x + half_w), min(1.0, center_y + half_h))


def hand_block(boxes, expand: float, anchor: str) -> np.ndarray:
    """手部区域 40 维；``anchor`` 决定"手部区域"怎么取（见 :data:`ANCHORS`）。

    - ``largest``：面积最大的单个 hand 框（v1 recipe 的行为）；
    - ``union``：全部 hand 框的并集（v2 recipe 的行为）；
    - ``union_tools``：hand 并集 ∪ 全部器械框（类 3~7）；
    - ``union_brush``：hand 并集 ∪ 刷类框（类 6~7，针对 ``short_brush_cleaning``）。
    """

    hands = [(cx, cy, w, h) for c, cx, cy, w, h in boxes if c == HAND_CLASS_ID]
    if not hands:
        return np.zeros(N_CLASSES * 5, dtype=np.float32)
    if anchor == "largest":
        rects = [max(hands, key=lambda b: b[2] * b[3])]
    elif anchor == "union":
        rects = hands
    elif anchor == "union_tools":
        rects = hands + [(cx, cy, w, h) for c, cx, cy, w, h in boxes if c in TOOL_CLASS_IDS]
    elif anchor == "union_brush":
        rects = hands + [(cx, cy, w, h) for c, cx, cy, w, h in boxes if c in BRUSH_CLASS_IDS]
    else:
        raise ValueError(f"未知锚点: {anchor!r}（可选 {ANCHORS}）")

    x1, y1, x2, y2 = _expand_union(rects, expand)

    rw, rh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
    feat = np.zeros((N_CLASSES, 5), dtype=np.float32)
    best = np.zeros(N_CLASSES, dtype=np.float32)
    for c, cx, cy, w, h in boxes:
        if not (0 <= c < N_CLASSES) or not (x1 <= cx <= x2 and y1 <= cy <= y2):
            continue
        area = w * h
        if area >= best[c]:
            best[c] = area
            feat[c] = (1.0, (cx - x1) / rw, (cy - y1) / rh, w / rw, h / rh)
    return feat.reshape(-1)


def build(raw, expand: float, anchor: str, with_global: bool) -> dict:
    pool = {}
    for stem, (frames, labels) in raw.items():
        matrix = np.stack([np.concatenate([global_block(b), hand_block(b, expand, anchor)])
                           if with_global else hand_block(b, expand, anchor)
                           for b in frames]).astype(np.float32)
        pool[stem] = (matrix, labels)
    return pool


def evaluate(pool, train_stems, test_stems, seed: int) -> tuple[float, float, dict]:
    """留出视频上的 (总体 acc, 全 idle 基线, 逐动作类召回)。"""

    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, early_stopping=True,
                                           validation_fraction=0.15, random_state=seed)
    model.fit(np.concatenate([pool[s][0] for s in train_stems]),
              np.concatenate([pool[s][1] for s in train_stems]))
    x_test = np.concatenate([pool[s][0] for s in test_stems])
    y_test = np.concatenate([pool[s][1] for s in test_stems])
    predicted = model.predict(x_test)
    recall = {}
    for index, name in enumerate(ACTION_NAMES):
        mask = y_test == index
        recall[name] = 100.0 * float((predicted[mask] == index).mean()) if mask.any() else None
    return (100.0 * float((predicted == y_test).mean()),
            100.0 * float((y_test == 0).mean()), recall)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--held", type=int, default=6, help="每折留出的视频数（train 共 28）")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="数据集根（含 labels/train 与 frames/train）")
    parser.add_argument("--anchors", default=",".join(ANCHORS), help="参与筛选的锚点（逗号分隔）")
    parser.add_argument("--expands", default=",".join(str(v) for v in EXPANDS),
                        help="参与筛选的扩张倍数（逗号分隔）")
    parser.add_argument("--only-hand", action="store_true", help="只筛 仅手部 40 维契约（更快）")
    parser.add_argument("--also-global", action="store_true", help="只筛 全局+手部 80 维契约")
    args = parser.parse_args()
    from scipy.stats import wilcoxon

    anchors = tuple(a for a in args.anchors.split(",") if a)
    expands = tuple(float(v) for v in args.expands.split(",") if v)
    print("[筛] 读取 auto train 原始框与标签 …", flush=True)
    raw = read_train(Path(args.root))
    stems = sorted(raw)
    print(f"[筛] {len(stems)} 视频 / {sum(len(v[1]) for v in raw.values())} 帧；"
          f"{args.folds} 折 × 留出 {args.held} 视频；锚点 {anchors}；扩张 {expands}", flush=True)

    splits = []
    for seed in range(args.folds):
        rng = np.random.default_rng(seed)
        order = list(stems)
        rng.shuffle(order)
        splits.append((order[args.held:], order[:args.held]))

    combos = []
    for with_global in (True, False):
        if args.only_hand and with_global:
            continue
        if args.also_global and not with_global:
            continue
        for anchor in anchors:
            for expand in expands:
                combos.append((with_global, anchor, expand))

    results = {}
    for with_global, anchor, expand in combos:
        pool = build(raw, expand, anchor, with_global)
        dim = next(iter(pool.values()))[0].shape[1]
        deltas, recalls = [], {name: [] for name in ACTION_NAMES}
        for tr, te in splits:
            acc, idle, recall = evaluate(pool, tr, te, seed=0)
            deltas.append(acc - idle)
            for name, value in recall.items():
                if value is not None:
                    recalls[name].append(value)
        name = f"{'全局+手部' if with_global else '仅手部'} expand={expand} anchor={anchor}"
        results[name] = (np.array(deltas), dim, {k: np.median(v) if v else None for k, v in recalls.items()})
        print(f"  {name:<40} 维 {dim:>4}  Δ中位 {np.median(deltas):+6.2f}", flush=True)

    ref_name = "全局+手部 expand=1.5 anchor=largest"
    ref = results[ref_name][0] if ref_name in results else next(iter(results.values()))[0]
    print(f"\n{'变体':<42}{'维度':>6}{'Δ中位':>9}   配对 vs 现行(1.5×最大框)")
    print("-" * 96)
    for name, (delta, dim, _recall) in sorted(results.items(), key=lambda kv: -np.median(kv[1][0])):
        diff = delta - ref
        nz = diff[diff != 0]
        p = wilcoxon(nz).pvalue if len(nz) else float("nan")
        print(f"{name:<42}{dim:>6}{np.median(delta):>+9.2f}   "
              f"{np.median(diff):+6.2f}pp {int((diff > 0).sum())}/{int((diff < 0).sum())} p={p:.3f}")

    print(f"\n逐动作类召回（留出视频，跨折中位 %；'—' = 该折无正样本）：")
    header = f"{'变体':<42}" + "".join(f"{n[:10]:>12}" for n in ACTION_NAMES)
    print(header)
    print("-" * len(header))
    for name, (_delta, _dim, recall) in sorted(results.items(), key=lambda kv: -np.median(kv[1][0])):
        cells = "".join(f"{(f'{recall[n]:.1f}' if recall[n] is not None else '—'):>12}" for n in ACTION_NAMES)
        print(f"{name:<42}{cells}")

    print("\n判读：源域筛选（只碰 train）。Δ = 留出视频逐帧 acc − 该折全 idle 基线（帧级 micro）。"
          "\n稀类（flush / sbc）召回看逐类表：总体 Δ 会被 idle/insert 稀释。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
