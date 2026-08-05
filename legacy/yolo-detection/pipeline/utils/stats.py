#!/usr/bin/env python3
"""
数据集生成后的样本分布统计(训练帧粒度)。

纯扫描落盘的 label 文件(datasets/<组>/labels/{train,val}/*.txt),不解码视频、不需 torch,
因此统计是"生成完毕的数据集"的纯函数:可信、可随时独立重算(不必重建)。

  - 帧数 = label 文件数(一帧一文件);框数 = 所有文件的行数之和;class_id = 每行首列。
  - 逐类给出 train/val 帧数、框数与 val 帧占比,并对空 split / 某类 val 无样本给出提示
    (与 04_validate 的验收口径呼应)。

独立重算:
    python3 -c "from utils import stats; stats.main()"
"""
from collections import defaultdict

from utils.common import ROOT, load_config

DATASETS = ROOT / "datasets"
SPLITS = ("train", "val")


def scan_group(group_dir):
    """扫描一个组的 labels/,返回 {split: {frames, boxes, cls_frames{cid}, cls_boxes{cid}}}。"""
    out = {}
    for split in SPLITS:
        d = group_dir / "labels" / split
        frames = boxes = 0
        cls_frames = defaultdict(int)
        cls_boxes = defaultdict(int)
        for txt in sorted(d.glob("*.txt")) if d.exists() else []:
            frames += 1
            seen = set()
            for line in txt.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                cid = int(line.split()[0])
                boxes += 1
                cls_boxes[cid] += 1
                seen.add(cid)
            for cid in seen:
                cls_frames[cid] += 1
        out[split] = {"frames": frames, "boxes": boxes,
                      "cls_frames": cls_frames, "cls_boxes": cls_boxes}
    return out


def print_distribution(group, class_names, group_dir=None):
    """打印一个组的逐类 train/val 帧/框分布 + 提示。返回扫描结果 dict。"""
    group_dir = group_dir or (DATASETS / group)
    st = scan_group(group_dir)
    tr, va = st["train"], st["val"]

    print(f"\n=== 样本分布 · {group}  (训练帧粒度,扫描 {group_dir}/labels) ===")
    print(f"{'类别':<22} {'train帧':>8} {'val帧':>7} {'train框':>8} {'val框':>7} {'val帧占比':>9}")
    print("-" * 66)
    for cid, name in enumerate(class_names):
        tf, vf = tr["cls_frames"].get(cid, 0), va["cls_frames"].get(cid, 0)
        tb, vb = tr["cls_boxes"].get(cid, 0), va["cls_boxes"].get(cid, 0)
        ratio = (vf / (tf + vf)) if (tf + vf) else 0.0
        print(f"{name:<22} {tf:>8} {vf:>7} {tb:>8} {vb:>7} {ratio:>8.0%}")
    tf_all, vf_all = tr["frames"], va["frames"]
    tb_all, vb_all = tr["boxes"], va["boxes"]
    ratio_all = (vf_all / (tf_all + vf_all)) if (tf_all + vf_all) else 0.0
    print("-" * 66)
    print(f"{'合计(帧/框)':<22} {tf_all:>8} {vf_all:>7} {tb_all:>8} {vb_all:>7} {ratio_all:>8.0%}")

    # 提示:与验收口径呼应
    warns = []
    if tf_all == 0:
        warns.append("train 为空 —— 无法训练")
    if vf_all == 0:
        warns.append("val 为空 —— 无法验证/验收(04 会判 FAIL)")
    for cid, name in enumerate(class_names):
        tf, vf = tr["cls_frames"].get(cid, 0), va["cls_frames"].get(cid, 0)
        if tf + vf == 0:
            warns.append(f"类别 {name} 训练/验证都无样本")
        elif vf == 0:
            warns.append(f"类别 {name} val 无样本 —— 该类无法评估(04 会判 FAIL)")
    for w in warns:
        print(f"  ⚠️ {w}")
    return st


def main():
    cfg = load_config()
    groups = cfg["groups"]
    if not DATASETS.exists():
        raise SystemExit(f"未找到 {DATASETS},先跑 02_build_dataset.py")
    for g, class_names in groups.items():
        gdir = DATASETS / g
        if not gdir.exists():
            print(f"\n[skip] {g}: 无 {gdir}(尚未生成)")
            continue
        print_distribution(g, class_names, gdir)


if __name__ == "__main__":
    main()
