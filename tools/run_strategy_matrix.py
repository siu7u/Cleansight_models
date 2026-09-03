"""四策略 × 多 seed 一键对照矩阵（特征提取范围实验的正式复跑工具）。

以框架 CLI（cli.train / benchmark.cli.eval）为唯一执行入口，按健康配方
（weight_decay + dropout + 早停 + 段级 best 指标，见 docs/FEATURE_STRATEGY_COMPARE.md
坍缩分析）训练指定策略列表 × seed 列表，对每个 run 自动跑正式评估，并汇总
每策略的逐 seed 指标与中位数，产出 <runs-dir>/STRATEGY_SUMMARY.md。

用法：

    python tools/run_strategy_matrix.py --runs-dir runs/strategy_compare
    # 默认四策略 × seed 42/7/2026；可用 --strategies/--seeds 裁剪

本脚本只做编排（shell 到框架 CLI），不重实现训练/评估逻辑。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PY = REPO_ROOT / "../CleanSightBackend/.venv/bin/python"
# 策略名 → 实验配置（特征提取范围矩阵：bbox 编码固定，只变提取范围 + ROI 网格）
STRATEGIES: dict[str, str] = {
    "bbox-40-global": "gru-actionmixed-auto.yaml",
    "bbox-40-hand": "gru-actionmixed-auto-hand.yaml",
    "bbox-80-global-hand": "gru-actionmixed-auto-global-hand.yaml",
    "roi-grid-144": "gru-actionmixed-auto-roi.yaml",
}
DEFAULT_SEEDS = (42, 7, 2026)
# 健康配方默认值（2026-09-03 坍缩诊断的代码层修复落地后的推荐配方）
DEFAULT_WD = 0.0001
DEFAULT_DROPOUT = 0.2
DEFAULT_PATIENCE = 4
DEFAULT_EPOCHS = 20
DEFAULT_BEST_METRIC = "val_f1_0.5"  # 段级指标，避免 val_acc 偏爱 idle 坍缩解


def run_cli(args: list[str]) -> None:
    proc = subprocess.run(
        [str(BACKEND_PY), *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        errors="replace",
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"命令失败: {' '.join(args[:6])}… exit={proc.returncode}")


def train_config(strategy_cfg: str, seed: int, runs_dir: Path, recipe: list[str]) -> str:
    """训练单个策略×seed，返回 run 目录路径（从 stdout 的 run_dir= 行解析）。"""

    proc = subprocess.run(
        [str(BACKEND_PY), "-m", "framework.cleansight_eval.cli.train",
         "--config", f"framework/experiments/{strategy_cfg}",
         "--runs-dir", str(runs_dir), "--seed", str(seed), *recipe],
        cwd=REPO_ROOT, text=True, capture_output=True, errors="replace",
    )
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"训练失败 cfg={strategy_cfg} seed={seed}")
    for line in proc.stdout.splitlines():
        if line.startswith("[train] run_dir="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"未解析到 run_dir: cfg={strategy_cfg} seed={seed}")


def config_file_of_run(run_dir: Path) -> str:
    """从 config.resolved.json 的溯源取回源配置文件（train 时存的绝对路径）。"""

    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    source = (cfg.get("_config_provenance") or {}).get("source_path")
    if not source:
        raise SystemExit(f"{run_dir} 缺少 _config_provenance.source_path")
    return source


def evaluate_run(run_dir: Path) -> Path:
    """对 run 的 best.pt 跑正式评估，返回 evaluation.json 路径。"""

    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"{run_dir} 缺少 best.pt")
    run_cli([
        "-m", "benchmark.cli.eval",
        "--config", config_file_of_run(run_dir),
        "--ckpt", str(ckpt),
    ])
    evals = sorted((run_dir / "evals").glob("*.evaluation.json"))
    if not evals:
        raise SystemExit(f"{run_dir} 评估未产出 evaluation.json")
    return evals[-1]


def summarize(runs_dir: Path, rows: list[dict]) -> str:
    """按策略聚合逐 seed 指标与中位数，返回 markdown 摘要文本。"""

    lines = [
        "# 特征策略对照矩阵汇总（一键复跑）",
        "",
        f"- 配方：weight_decay={DEFAULT_WD} / dropout={DEFAULT_DROPOUT} / "
        f"patience={DEFAULT_PATIENCE} / best_metric={DEFAULT_BEST_METRIC}",
        f"- 数据：v3（含 task#204 修正）；test 锚定 task#195/#199（仅 idle/insert/withdraw）",
        f"- 运行目录：`{runs_dir}`",
        "",
        "| 策略 | seed | acc | edit | F1@0.1 | F1@0.25 | 非idle预测帧 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    per_strategy: dict[str, list[dict]] = {}
    for row in rows:
        per_strategy.setdefault(row["strategy"], []).append(row)
    for strategy in sorted(per_strategy):
        items = sorted(per_strategy[strategy], key=lambda r: r["seed"])
        for row in items:
            lines.append(
                f"| {strategy} | {row['seed']} | {row['acc']:.2f} | {row['edit']:.2f} "
                f"| {row['f1_01']:.2f} | {row['f1_025']:.2f} | {row['nonidle']} |"
            )
        def median(key):
            values = [row[key] for row in items]
            return statistics.median(values)
        lines.append(
            f"| **{strategy} 中位数** | — | {median('acc'):.2f} | {median('edit'):.2f} "
            f"| {median('f1_01'):.2f} | {median('f1_025'):.2f} | — |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="四策略 × 多 seed 一键对照矩阵")
    parser.add_argument("--runs-dir", default="runs/strategy_compare")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WD)
    parser.add_argument("--best-metric", default=DEFAULT_BEST_METRIC)
    parser.add_argument("--skip-train", action="store_true", help="只评估+汇总已有 run")
    parser.add_argument("--skip-eval", action="store_true", help="只训练+汇总（跳过评估）")
    args = parser.parse_args(argv)

    if not BACKEND_PY.is_file():
        raise SystemExit(f"backend venv python 不存在: {BACKEND_PY}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    strategies = [s for s in args.strategies.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        raise SystemExit(f"未知策略 {unknown}；可选: {sorted(STRATEGIES)}")
    runs_dir = (REPO_ROOT / args.runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    recipe = [
        "-S", f"train.weight_decay={args.weight_decay}",
        "-S", f"model.dropout={args.dropout}",
        "-S", f"train.patience={args.patience}",
        "-S", f"train.epochs={args.epochs}",
        "-S", f"train.best_metric={args.best_metric}",
    ]

    run_dirs: list[tuple[str, int, Path]] = []
    if not args.skip_train:
        for strategy in strategies:
            for seed in seeds:
                cfg = STRATEGIES[strategy]
                print(f"== 训练 {strategy} seed={seed} ==", flush=True)
                run_dir = Path(train_config(cfg, seed, runs_dir, recipe))
                run_dirs.append((strategy, seed, run_dir))
    else:
        for run_dir in sorted(runs_dir.glob("gru-*")):
            cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
            mapping = cfg.get("feature_schema", {}).get("version", "?")
            run_dirs.append((mapping, -1, run_dir))

    rows: list[dict] = []
    if not args.skip_eval:
        for strategy, seed, run_dir in run_dirs:
            print(f"== 评估 {run_dir.name} ==", flush=True)
            eval_path = evaluate_run(run_dir)
            data = json.loads(eval_path.read_text(encoding="utf-8"))
            summary = data["metrics"]["summary"]
            seg = data["metrics"]["details"]["temporal"]["segment"]
            cm = data["metrics"]["details"]["temporal"]["frame"]["confusion_matrix_rows_truth_cols_prediction"]
            rows.append({
                "strategy": strategy if strategy != "?" else Path(config_file_of_run(run_dir)).stem,
                "seed": seed,
                "acc": summary["acc"]["value"],
                "edit": summary["edit"]["value"],
                "f1_01": seg["f1_at_iou"]["0.10"] * 100,
                "f1_025": seg["f1_at_iou"]["0.25"] * 100,
                "nonidle": sum(sum(cm[r][c] for r in range(len(cm))) for c in range(1, len(cm[0]))),
            })

    text = summarize(runs_dir, rows)
    print(text)
    (runs_dir / "STRATEGY_SUMMARY.md").write_text(text, encoding="utf-8")
    print(f"摘要已写入 {runs_dir / 'STRATEGY_SUMMARY.md'}")


if __name__ == "__main__":
    sys.exit(main())
