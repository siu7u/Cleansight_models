#!/usr/bin/env python3
"""
在验证集上评测各组权重,输出逐类指标 + 按 config.acceptance 判 PASS/FAIL,并写验收报告。
有任一组 FAIL 时进程退出码非零(便于交付卡口 / CI)。
同时写入 timestamp 归档报告,避免覆盖历史验收结果。

需 torch + ultralytics —— 用本项目 .venv/bin/python(见 requirements.txt)。

用法(在 yolo_pipeline/ 下执行):
    <py> 04_validate.py                # 全部有权重的组
    <py> 04_validate.py group2_small   # 只验某组
    <py> 04_validate.py --split test   # 在 holdout test 上评估
    <py> 04_validate.py group1_large --weights versioned_weights/yolo-large-v2/best.pt
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys

from utils.common import ROOT, load_config

DATASETS = ROOT / "datasets"
RUNS = ROOT / "runs"


def timestamp() -> str:
    """返回适合文件名使用的验证报告时间戳。"""

    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_args():
    """解析要验证的分组、数据 split 和可选权重路径。"""

    parser = argparse.ArgumentParser(description="Validate grouped YOLO detectors.")
    parser.add_argument("groups", nargs="*", help="只验证指定分组;不填则验证全部有权重的组")
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="评估数据 split。val 用于训练后验收,test 用于最终 holdout 评估。",
    )
    parser.add_argument(
        "--weights",
        help="指定单个分组要评估的权重路径;不填则使用 runs/<组>/weights/best.pt。",
    )
    return parser.parse_args()


def evaluate(group, thr, split, weight_override=None):
    """跑指定 split,返回 (metrics_dict, per_class_list, passed, reasons, artifacts)。"""
    from ultralytics import YOLO

    data = DATASETS / group / "data.yaml"
    weight = weight_override or (RUNS / group / "weights" / "best.pt")
    weight = weight if weight.is_absolute() else ROOT / weight
    if not weight.exists():
        return None, None, None, [f"缺权重 {weight},先跑 03_train.py"], {}
    if not data.exists():
        return None, None, None, [f"缺 data.yaml {data},先跑 02_build_dataset.py"], {}

    model = YOLO(str(weight))
    m = model.val(data=str(data), split=split, verbose=False,
                  project=str(RUNS), name=f"{group}_{split}", exist_ok=True,
                  save_json=True)
    names = model.names
    box = m.box
    save_dir = Path(getattr(m, "save_dir", RUNS / f"{group}_{split}"))
    prediction_candidates = sorted(save_dir.glob("*.json")) if save_dir.exists() else []
    artifacts = {}
    if prediction_candidates:
        artifacts["predictions"] = str(prediction_candidates[-1])

    overall = {
        "map50": float(box.map50), "map50_95": float(box.map),
        "precision": float(box.mp), "recall": float(box.mr),
    }
    per_class = []
    for i, cidx in enumerate(list(box.ap_class_index)):
        per_class.append({
            "name": names[int(cidx)],
            "precision": float(box.p[i]), "recall": float(box.r[i]),
            "map50": float(box.ap50[i]),
        })

    reasons = []
    if overall["map50"] < thr["overall_map50"]:
        reasons.append(f"整体 mAP50 {overall['map50']:.3f} < {thr['overall_map50']}")
    if overall["map50_95"] < thr["overall_map50_95"]:
        reasons.append(f"整体 mAP50-95 {overall['map50_95']:.3f} < {thr['overall_map50_95']}")
    labeled = {pc["name"] for pc in per_class}
    for name in names.values():
        if name not in labeled:
            reasons.append(f"类别 {name} 验证集无样本/未检出(无法评估)")
    for pc in per_class:
        if pc["recall"] < thr["per_class_recall"]:
            reasons.append(f"{pc['name']} recall {pc['recall']:.3f} < {thr['per_class_recall']}")
        if pc["precision"] < thr["per_class_precision"]:
            reasons.append(f"{pc['name']} precision {pc['precision']:.3f} < {thr['per_class_precision']}")

    return overall, per_class, (len(reasons) == 0), reasons, artifacts


def write_report(group, split, checkpoint, overall, per_class, passed, reasons, thr, artifacts=None):
    """写入当前验收报告,并额外保存一份带时间戳的归档报告。"""

    artifacts = artifacts or {}
    lines = [f"# 验收报告 · {group}", "",
             f"数据集 split: `{split}`", "",
             f"权重: `{checkpoint}`", "",
             f"结论: **{'PASS ✅' if passed else 'FAIL ❌'}**", "",
             "## 整体指标", "",
             "| 指标 | 值 | 门槛 |", "|------|----|----|",
             f"| mAP@0.5 | {overall['map50']:.3f} | ≥ {thr['overall_map50']} |",
             f"| mAP@0.5:0.95 | {overall['map50_95']:.3f} | ≥ {thr['overall_map50_95']} |",
             f"| 平均 precision | {overall['precision']:.3f} | — |",
             f"| 平均 recall | {overall['recall']:.3f} | — |", "",
             "## 逐类指标", "",
             "| 类别 | precision | recall | mAP@0.5 |", "|------|-----------|--------|---------|"]
    for pc in per_class:
        lines.append(f"| {pc['name']} | {pc['precision']:.3f} | {pc['recall']:.3f} | {pc['map50']:.3f} |")
    lines += ["", f"门槛:逐类 recall ≥ {thr['per_class_recall']}、precision ≥ {thr['per_class_precision']}", ""]
    if reasons:
        lines += ["## 未达标项", ""] + [f"- {r}" for r in reasons]
    else:
        lines += ["全部达标。"]
    if artifacts.get("predictions"):
        lines += ["", "## 预测 Artifact", "", f"- 逐图预测: `{artifacts['predictions']}`"]
    out_name = "acceptance_report.md" if split == "val" else f"acceptance_report_{split}.md"
    out = RUNS / group / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")

    archive = RUNS / group / "reports" / f"acceptance_report_{split}-{timestamp()}.md"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text(text, encoding="utf-8")
    return out, archive


def main():
    args = parse_args()
    cfg = load_config()
    thr = cfg["acceptance"]
    if args.groups:
        groups = args.groups
    elif RUNS.exists():
        groups = [p.name for p in sorted(RUNS.iterdir()) if p.is_dir()]
    else:
        groups = []
    if not groups:
        raise SystemExit("没有可验证的组(先 03_train.py),或显式传组名。")
    if args.weights and len(groups) != 1:
        raise SystemExit("--weights 只能在验证单个分组时使用,例如: 04_validate.py group1_large --weights ...")

    weight_override = None
    if args.weights:
        weight_override = Path(args.weights)

    any_fail = False
    for g in groups:
        print(f"\n=== 验证 {g} split={args.split} ===")
        checkpoint = weight_override or (RUNS / g / "weights" / "best.pt")
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        overall, per_class, passed, reasons, artifacts = evaluate(g, thr, args.split, weight_override)
        if overall is None:
            print(f"  [skip] {'; '.join(reasons)}")
            any_fail = True
            continue
        print(f"  整体 mAP50={overall['map50']:.3f}  mAP50-95={overall['map50_95']:.3f}  "
              f"P={overall['precision']:.3f}  R={overall['recall']:.3f}")
        for pc in per_class:
            print(f"    {pc['name']:22s} P={pc['precision']:.3f} R={pc['recall']:.3f} "
                  f"mAP50={pc['map50']:.3f}")
        report, archive = write_report(g, args.split, checkpoint, overall, per_class, passed, reasons, thr, artifacts)
        print(f"  结论: {'PASS ✅' if passed else 'FAIL ❌'}   报告: {report}")
        print(f"  归档报告: {archive}")
        if artifacts.get("predictions"):
            print(f"  逐图预测 artifact: {artifacts['predictions']}")
        if not passed:
            any_fail = True
            for r in reasons:
                print(f"      - {r}")

    if any_fail:
        print("\n有组未通过验收(退出码 2)。")
        sys.exit(2)
    print("\n全部通过验收 ✅")


if __name__ == "__main__":
    main()
