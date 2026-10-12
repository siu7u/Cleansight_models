"""检测侧「置信度阈值 → 逐类召回」探针：判断某类漏检是**阈值损失**还是**权重能力不足**。

## 为什么需要它

自动标注链路（`cli.annotate` + `framework/experiments/auto-annotate*.yaml`）用**全局**
`conf` 阈值把 YOLO 输出筛成时序特征（`frames/<split>/*.txt`）。于是"某个动作类的检测证据
在时序侧恒为零"有两种完全不同的成因，处置方式也完全不同：

- **阈值太高**：降低该类阈值即可立刻找回证据（零训练成本）；
- **权重不认这个类**：阈值降到 0.01 也依然为零，只能靠重训/补标注，降阈值只会放噪声进来。

本探针一次前向（`conf=0.001` 取全部原始检测，含分数）拿到**整条阈值曲线**，再离线套阈值，
因此「阈值损失」与「能力不足」可以直接分开看，并且能读出边际收益开始衰减的拐点。

## 口径（与 mAP 不同，别混用）

召回按**贪心匹配**计算：同类、IoU ≥ `--iou-match`、每个 GT 只被一个预测匹配。
这测的是"给定阈值下还剩多少 GT 被框住"，不是 PR 曲线下的面积；与 ultralytics 的
`model.val()` 读数**不可直接比较**。

用法（仓库根执行）::

    python tools/probe_detection_conf_recall.py <weights.pt> --group group2_small --split test val
    python tools/probe_detection_conf_recall.py <weights.pt> --group group2_small --split val --out tmp/conf.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework"))

from cleansight_eval.detection.yolo import get_adapter  # noqa: E402

THRESHOLDS = (0.01, 0.05, 0.10, 0.15, 0.25, 0.40, 0.50)


def load_gt(labels_dir: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    """读 YOLO 标签目录 → ``{文件名去扩展名: [(cls, cx, cy, w, h), ...]}``（归一化 xywh）。

    键统一去扩展名：标注文件是 ``<stem>.txt``，而 ``predict`` 返回的 items 键是 ``<stem>.jpg``。
    """

    out: dict[str, list[tuple[int, float, float, float, float]]] = {}
    for p in labels_dir.glob("*.txt"):
        rows = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()[:5]
            rows.append((int(c), float(cx), float(cy), float(w), float(h)))
        out[p.stem] = rows
    return out


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """两个归一化 xywh 框的 IoU。"""

    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def sweep_split(weights: str, data_yaml: Path, labels_dir: Path, split: str,
                imgsz: int, device: str, iou_match: float) -> tuple[dict, dict]:
    """在一个 split 上算逐类召回 vs 阈值。返回 ``(阈值表, 类别名表)``。"""

    adapter = get_adapter("yolo")
    # ultralytics 会往 stdout 打进度条；重定向掉，只保留本探针自己的输出
    with contextlib.redirect_stdout(io.StringIO()):
        pred = adapter.predict(weights, str(data_yaml), split, imgsz, device,
                               conf=0.001, iou=0.7, max_det=300, agnostic_nms=False)
    names = {int(k): v for k, v in pred["labels"].items()}
    items = {Path(k).stem: v["predictions"] for k, v in pred["items"].items()}
    gt = load_gt(labels_dir)

    table: dict[str, dict] = {}
    for th in THRESHOLDS:
        hit = {n: 0 for n in names.values()}
        tot = {n: 0 for n in names.values()}
        for fname, rows in gt.items():
            boxes = [(p["class_id"], *p["xywhn"]) for p in items.get(fname, [])
                     if p["confidence"] >= th]
            used = [False] * len(boxes)
            for c, cx, cy, w, h in rows:
                tot[names[c]] += 1
                best, best_iou = -1, 0.0
                for i, (pc, *box) in enumerate(boxes):
                    if used[i] or pc != c:
                        continue
                    v = iou((cx, cy, w, h), box)
                    if v > best_iou:
                        best, best_iou = i, v
                if best >= 0 and best_iou >= iou_match:
                    used[best] = True
                    hit[names[c]] += 1
        table[f"{th:.2f}"] = {
            "per_class_recall": {n: (round(hit[n] / tot[n], 4) if tot[n] else None)
                                 for n in names.values()},
            "gt_instances": {n: tot[n] for n in names.values()},
        }
    return table, names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("weights", help="YOLO 权重路径")
    ap.add_argument("--group", default="group2_small",
                    help="检测分组目录名（datasets/cleansight-yolo/<group>）")
    ap.add_argument("--split", nargs="*", default=["test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--iou-match", type=float, default=0.5, help="召回的 IoU 匹配门槛")
    ap.add_argument("--out", default=None, help="可选：把逐类召回表写成 JSON")
    args = ap.parse_args()

    group_dir = ROOT / "datasets" / "cleansight-yolo" / args.group
    data_yaml = group_dir / "data.yaml"
    if not data_yaml.is_file():
        print(f"[ERROR] 缺少 {data_yaml}", file=sys.stderr)
        return 1

    result = {"weights": str(Path(args.weights).resolve()), "group": args.group,
              "iou_match": args.iou_match, "splits": {}}
    for split in args.split:
        table, names = sweep_split(str(Path(args.weights).resolve()), data_yaml,
                                   group_dir / "labels" / split, split,
                                   args.imgsz, args.device, args.iou_match)
        result["splits"][split] = table
        print(f"\n=== {args.group} / {split} @ imgsz {args.imgsz} @ IoU {args.iou_match} ===")
        print(f"{'conf':>5}  " + "".join(f"{n[:14]:>16}" for n in names.values()))
        for th, row in table.items():
            cells = ""
            for n in names.values():
                value = row["per_class_recall"][n]
                text = "n/a" if value is None else f"{value:.4f}"
                cells += f"{text:>16}"
            print(f"{th:>5}  {cells}")
        print("  GT 实例数: " + ", ".join(f"{n}={table['0.25']['gt_instances'][n]}"
                                        for n in names.values()))

    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
