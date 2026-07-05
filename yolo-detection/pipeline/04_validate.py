#!/usr/bin/env python3
"""
在验证集上评测各组权重,输出逐类指标 + 按 config.acceptance 判 PASS/FAIL,并写验收报告。
有任一组 FAIL 时进程退出码非零(便于交付卡口 / CI)。

需 torch + ultralytics —— 用本项目 .venv/bin/python(见 requirements.txt)。

用法(在 yolo_pipeline/ 下执行):
    <py> 04_validate.py                # 全部有权重的组
    <py> 04_validate.py group2_small   # 只验某组
"""
import sys

from utils.common import ROOT, load_config

DATASETS = ROOT / "datasets"
RUNS = ROOT / "runs"


def evaluate(group, thr):
    """跑 val,返回 (metrics_dict, per_class_list, passed, reasons)。"""
    from ultralytics import YOLO

    data = DATASETS / group / "data.yaml"
    weight = RUNS / group / "weights" / "best.pt"
    if not weight.exists():
        return None, None, None, [f"缺权重 {weight},先跑 03_train.py"]
    if not data.exists():
        return None, None, None, [f"缺 data.yaml {data},先跑 02_build_dataset.py"]

    model = YOLO(str(weight))
    m = model.val(data=str(data), split="val", verbose=False,
                  project=str(RUNS), name=f"{group}_val", exist_ok=True)
    names = model.names
    box = m.box

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

    return overall, per_class, (len(reasons) == 0), reasons


def write_report(group, overall, per_class, passed, reasons, thr):
    lines = [f"# 验收报告 · {group}", "",
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
    out = RUNS / group / "acceptance_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main():
    cfg = load_config()
    thr = cfg["acceptance"]
    requested = [a for a in sys.argv[1:] if not a.startswith("-")]
    if requested:
        groups = requested
    elif RUNS.exists():
        groups = [p.name for p in sorted(RUNS.iterdir()) if p.is_dir()]
    else:
        groups = []
    if not groups:
        raise SystemExit("没有可验证的组(先 03_train.py),或显式传组名。")

    any_fail = False
    for g in groups:
        print(f"\n=== 验证 {g} ===")
        overall, per_class, passed, reasons = evaluate(g, thr)
        if overall is None:
            print(f"  [skip] {'; '.join(reasons)}")
            any_fail = True
            continue
        print(f"  整体 mAP50={overall['map50']:.3f}  mAP50-95={overall['map50_95']:.3f}  "
              f"P={overall['precision']:.3f}  R={overall['recall']:.3f}")
        for pc in per_class:
            print(f"    {pc['name']:22s} P={pc['precision']:.3f} R={pc['recall']:.3f} "
                  f"mAP50={pc['map50']:.3f}")
        report = write_report(g, overall, per_class, passed, reasons, thr)
        print(f"  结论: {'PASS ✅' if passed else 'FAIL ❌'}   报告: {report}")
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
