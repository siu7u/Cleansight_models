"""小目标逐类阈值分析与淘汰决策 CLI：python -m benchmark.cli.analyze。

对 group2_small（或任何检测组）的 best.pt 做精细分析：
  1. 逐类 P/R/mAP50，标出 P/R 低于阈值的类别（复用 framework ``DetectionPipeline.predict``
     的 native_metrics，benchmark 不直接调 adapter）
  2. 多 conf 阈值扫描，观察各类别 recall 是否有挽救空间
  3. 输出 保留/边界/淘汰 三组决策
  4. 对淘汰类调用 framework ``detection.data_tools.build_trimmed_dataset`` 生成裁剪数据集

用法:
    python -m benchmark.cli.analyze --config framework/experiments/yolo-clean-small.yaml \
        --ckpt runs/<run>/checkpoints/group2_small/weights/best.pt [--threshold 0.3]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from framework.cleansight_eval.core.config import load_config
from framework.cleansight_eval.core.environment import pick_device
from framework.cleansight_eval.core.registry import get_pipeline
from benchmark.core.analysis import classify_classes

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "runs" / "small_analysis"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="实验配置 YAML（如 framework/experiments/yolo-clean-small.yaml）")
    p.add_argument("--ckpt", required=True, help="best.pt 路径")
    p.add_argument("--threshold", type=float, default=0.3, help="逐类 P/R 淘汰阈值(默认 0.3)")
    p.add_argument(
        "--conf-thresholds", type=float, nargs="*",
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        help="分析的 conf 阈值列表",
    )
    p.add_argument("--trim-dir", default=None,
                   help="裁剪数据集输出目录(默认 datasets/cleansight-yolo/<group>_kept)")
    return p.parse_args(argv)


def analyze_per_class(cfg: dict, ckpt: str, device, conf: float = 0.001) -> dict:
    """通过 DetectionPipeline.predict 拿 native_metrics 的逐类指标。"""

    cfg = dict(cfg)
    evaluation = dict(cfg.get("evaluation", {}))
    evaluation["conf"] = conf
    cfg["evaluation"] = evaluation
    pipeline = get_pipeline(cfg["pipeline"])
    output = pipeline.predict(cfg, ckpt, device)
    native = output.native_metrics or {}
    per_class: Dict[str, dict] = {}
    for name, metrics in (native.get("per_class") or {}).items():
        per_class[str(name)] = {
            "precision": round(float(metrics.get("precision", 0)), 4),
            "recall": round(float(metrics.get("recall", 0)), 4),
            "map50": round(float(metrics.get("map50", 0)), 4),
        }
    for name in (native.get("names") or {}).values():
        if str(name) not in per_class:
            per_class[str(name)] = {
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "note": "验证集无检出/无样本",
            }
    return {
        "map50": round(float(native.get("map50", 0)), 4),
        "map50_95": round(float(native.get("map50_95", 0)), 4),
        "precision": round(float(native.get("precision", 0)), 4),
        "recall": round(float(native.get("recall", 0)), 4),
        "names": native.get("names", {}),
        "per_class": per_class,
    }


def analyze_conf_sweep(cfg: dict, ckpt: str, device, conf_thresholds: List[float]) -> List[dict]:
    """在不同 conf 阈值下评测，观察各类别 recall 变化。"""

    results = []
    for conf in conf_thresholds:
        analysis = analyze_per_class(cfg, ckpt, device, conf=conf)
        entry = {
            "conf": conf,
            "map50": analysis["map50"],
            "precision": analysis["precision"],
            "recall": analysis["recall"],
            "per_class": {
                name: {"precision": m.get("precision", 0), "recall": m.get("recall", 0)}
                for name, m in analysis["per_class"].items()
            },
        }
        results.append(entry)
    return results


def generate_report(
    analysis: dict,
    sweep: List[dict],
    keep: List[str],
    borderline: List[str],
    eliminate: List[str],
    threshold: float,
) -> str:
    """生成 Markdown 分析报告。"""

    lines = [
        "# 逐类阈值分析报告",
        "",
        f"生成时间: {datetime.now().isoformat()}",
        f"淘汰阈值: P < {threshold} 且 R < {threshold}",
        "",
        "## 1. 决策",
        "",
        f"### ✅ 保留在 YOLO ({len(keep)} 类)",
    ]
    for c in keep:
        m = analysis["per_class"].get(c, {})
        lines.append(f"- **{c}**: P={m.get('precision', 0):.4f} R={m.get('recall', 0):.4f} "
                     f"mAP50={m.get('map50', 0):.4f}")

    lines += ["", f"### ⚠️ 边界 ({len(borderline)} 类) - 建议进一步优化"]
    for c in borderline:
        m = analysis["per_class"].get(c, {})
        lines.append(f"- **{c}**: P={m.get('precision', 0):.4f} R={m.get('recall', 0):.4f} "
                     f"mAP50={m.get('map50', 0):.4f}")

    lines += ["", f"### ❌ 淘汰 ({len(eliminate)} 类) - 转图像特征融合"]
    for c in eliminate:
        m = analysis["per_class"].get(c, {})
        note = m.get("note", "")
        lines.append(f"- **{c}**: P={m.get('precision', 0):.4f} R={m.get('recall', 0):.4f} "
                     f"mAP50={m.get('map50', 0):.4f} {note}")

    lines += [
        "",
        "## 2. 整体指标",
        "",
        "| 指标 | 值 |",
        "|------|----|",
        f"| mAP@0.5 | {analysis['map50']:.4f} |",
        f"| mAP@0.5:0.95 | {analysis['map50_95']:.4f} |",
        f"| Precision | {analysis['precision']:.4f} |",
        f"| Recall | {analysis['recall']:.4f} |",
        "",
        "## 3. 逐类详情",
        "",
        "| 类别 | Precision | Recall | mAP50 | 决策 |",
        "|------|----------:|-------:|------:|------|",
    ]
    for cls_name, m in analysis["per_class"].items():
        if cls_name in keep:
            decision = "✅ 保留"
        elif cls_name in borderline:
            decision = "⚠️ 边界"
        else:
            decision = "❌ 淘汰"
        lines.append(f"| {cls_name} | {m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | "
                     f"{m.get('map50', 0):.4f} | {decision} |")

    if sweep:
        lines += [
            "",
            "## 4. 置信度扫描",
            "",
            "在不同 conf 阈值下各类别 recall 变化（观察是否有挽救空间）：",
            "",
        ]
        confs = [s["conf"] for s in sweep]
        header = "| 类别 |" + "".join(f" conf={c:.2f} |" for c in confs)
        lines.append(header)
        lines.append("|------|" + "|".join(["------:" for _ in confs]) + "|")
        for cls_name in analysis["names"].values():
            row = f"| {cls_name} |"
            for s in sweep:
                pc = s["per_class"].get(str(cls_name), {})
                row += f" {pc.get('recall', 0):.4f} |"
            lines.append(row)

    lines += [
        "",
        "## 5. 后续行动",
        "",
        f"1. 用裁剪后的 data.yaml（仅保留 {len(keep)} 类）重训 YOLO",
        f"2. 对淘汰类（{', '.join(eliminate) if eliminate else '无'}）部署特征融合模型",
        f"3. 边界类（{', '.join(borderline) if borderline else '无'}）尝试 copy_paste/mosaic 增强 + 更高分辨率",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        print(f"[ERROR] 权重文件不存在: {ckpt}", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    device = pick_device()
    group = cfg["data"].get("name", "unknown")

    print(f"[analyze] 权重: {ckpt}")
    print(f"[analyze] 配置: {args.config}")
    print(f"[analyze] 淘汰阈值: P<{args.threshold} 或 R<{args.threshold}")

    # 1. 逐类分析
    print("\n=== 逐类分析 ===")
    analysis = analyze_per_class(cfg, str(ckpt), device)
    print(f"  整体: mAP50={analysis['map50']:.4f} P={analysis['precision']:.4f} "
          f"R={analysis['recall']:.4f}")
    for cls_name, m in analysis["per_class"].items():
        note = f" ⚠️ {m.get('note', '')}" if "note" in m else ""
        print(f"  {cls_name:<20} P={m.get('precision', 0):.4f} R={m.get('recall', 0):.4f} "
              f"mAP50={m.get('map50', 0):.4f}{note}")

    # 2. 分类决策
    keep, borderline, eliminate = classify_classes(analysis["per_class"], args.threshold)
    print(f"\n=== 分类决策 (阈值={args.threshold}) ===")
    print(f"  ✅ 保留 ({len(keep)}): {keep if keep else '(无)'}")
    print(f"  ⚠️ 边界 ({len(borderline)}): {borderline if borderline else '(无)'}")
    print(f"  ❌ 淘汰 ({len(eliminate)}): {eliminate if eliminate else '(无)'}")

    # 3. conf 扫描
    print(f"\n=== 置信度扫描 (conf={args.conf_thresholds}) ===")
    sweep = analyze_conf_sweep(cfg, str(ckpt), device, args.conf_thresholds)
    for s in sweep:
        print(f"  conf={s['conf']:.2f}: mAP50={s['map50']:.4f} P={s['precision']:.4f} "
              f"R={s['recall']:.4f}")

    # 4. 生成裁剪数据集
    trimmed_yaml = None
    all_keep = keep + borderline  # 边界类也先保留
    if eliminate:
        from framework.cleansight_eval.detection.data_tools import build_trimmed_dataset

        group_dir = REPO_ROOT / "datasets" / "cleansight-yolo" / group
        output_dir = Path(args.trim_dir) if args.trim_dir else group_dir.parent / f"{group}_kept"
        print(f"\n=== 生成裁剪数据集 (保留 {len(all_keep)} 类) ===")
        try:
            trimmed_yaml = build_trimmed_dataset(group_dir, all_keep, output_dir)
        except Exception as exc:
            print(f"  [ERROR] 裁剪失败: {exc}")
    else:
        print("\n无淘汰类别，跳过数据集裁剪")

    # 5. 保存报告
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_md = generate_report(analysis, sweep, keep, borderline, eliminate, args.threshold)
    md_path = REPORTS_DIR / f"analysis_{group}_{ts}.md"
    md_path.write_text(report_md, encoding="utf-8")

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "ckpt": str(ckpt),
        "config": args.config,
        "threshold": args.threshold,
        "analysis": analysis,
        "sweep": sweep,
        "keep": keep,
        "borderline": borderline,
        "eliminate": eliminate,
        "trimmed_yaml": str(trimmed_yaml) if trimmed_yaml else None,
    }
    json_path = REPORTS_DIR / f"analysis_{group}_{ts}.json"
    json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")

    print(f"\n报告已保存: {md_path}")
    print(f"          : {json_path}")

    if eliminate:
        print(f"\n⚠️  建议行动:")
        print(f"  1. 用裁剪数据集训练: "
              f"python -m framework.cleansight_eval.cli.sweep --group {group}_kept --preset small_s_960")
        print(f"  2. 对淘汰类 ({', '.join(eliminate)}) 训练特征融合: "
              f"python -m framework.cleansight_eval.cli.train --config framework/experiments/roi-fusion.yaml")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
