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
import re

import numpy as np
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_PY = REPO_ROOT / "../CleanSightBackend/.venv/bin/python"
# 策略名 → 实验配置（特征提取范围/编码矩阵：模型与超参固定，只变特征契约）
STRATEGIES: dict[str, str] = {
    "bbox-40-global": "gru-actionmixed-auto.yaml",
    "bbox-40-hand": "gru-actionmixed-auto-hand.yaml",
    "bbox-80-global-hand": "gru-actionmixed-auto-global-hand.yaml",
    "roi-grid-144": "gru-actionmixed-auto-roi.yaml",
    "roi-grid-96-v2": "gru-actionmixed-auto-roi-v2.yaml",  # 可见性重排（P2b）
    # 跨架构 / 跨 feed-mode 对照（同一 ROI-144 契约，全序列非因果模型无因果平滑）
    "mstcn-roi-144": "mstcn-actionmixed-auto-roi.yaml",
    "transformer-roi-144": "transformer-actionmixed-auto-roi.yaml",
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


def eval_config_of_run(run_dir: Path) -> Path:
    """评估用配置：优先 run 自己的 ``config.resolved.json``。

    它 = 源配置 + 全部 ``-S`` 覆盖（例如 ``model.normalization=zscore``）。评估必须用同一份，
    否则归一化 buffer 等结构差异会导致加载失败或静默按另一口径评估；实测与源配置评估结果
    逐位一致（2026-09-18 核对 acc/edit）。缺失时回退到源配置。
    """

    resolved = run_dir / "config.resolved.json"
    return resolved if resolved.is_file() else Path(config_file_of_run(run_dir))


def strategy_name_of_run(run_dir: Path) -> str:
    """反查 run 的策略名：优先按源配置文件名匹配 STRATEGIES，回退到 feature_mapping 版本。"""

    by_config = {cfg: name for name, cfg in STRATEGIES.items()}
    name = by_config.get(Path(config_file_of_run(run_dir)).name)
    if name:
        return name
    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    return str((cfg.get("feature_schema") or {}).get("version", "?"))


def seed_of_run(run_dir: Path) -> int:
    """从 env.json 读回训练 seed；读不到返回 -1（标记为未知）。"""

    env_path = run_dir / "env.json"
    if env_path.is_file():
        seed = (json.loads(env_path.read_text(encoding="utf-8")) or {}).get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            return seed
    return -1


def evaluate_run(run_dir: Path, *, reuse: bool = True, min_duration: int | None = None) -> Path:
    """对 run 的 best.pt 跑正式评估，返回 evaluation.json 路径。

    ``reuse=True``（默认）时若已有**同一口径**的评估产物就直接复用：汇总应当幂等，重复汇总
    不该反复重跑评估、在 ``evals/`` 里堆积重复 json；需要重跑传 ``reuse=False``。
    ``min_duration`` 非空时改用该平滑阈值的配置变体 + 独立输出目录，口径变体不与默认口径
    混用，也不污染 run 内 ``evals/``。
    """

    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.is_file():
        raise SystemExit(f"{run_dir} 缺少 best.pt")

    if min_duration is None:
        existing = sorted((run_dir / "evals").glob("*.evaluation.json"))
        if reuse and existing:
            print(f"  复用已有评估: {existing[-1].name}", flush=True)
            return existing[-1]
        run_cli([
            "-m", "benchmark.cli.eval",
            "--config", str(eval_config_of_run(run_dir)),
            "--ckpt", str(ckpt),
        ])
        evals = sorted((run_dir / "evals").glob("*.evaluation.json"))
        if not evals:
            raise SystemExit(f"{run_dir} 评估未产出 evaluation.json")
        return evals[-1]

    source = eval_config_of_run(run_dir)
    cfg_dir = run_dir.parent / "_eval_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    variant = cfg_dir / f"{source.stem}-md{min_duration}.json"
    # 用配置层的 apply_overrides 生成变体，而不是文本替换：源可能是 JSON
    # （run 的 config.resolved.json），文本插入会静默不生效、导致按默认口径评估。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from framework.cleansight_eval.core.config import apply_overrides, load_config

    variant.write_text(
        json.dumps(
            apply_overrides(load_config(source),
                            [("evaluation.smoothing_min_duration", min_duration)]),
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    out_dir = run_dir.parent / f"_eval_md{min_duration}" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.evaluation.json"))
    if reuse and existing:
        print(f"  复用已有评估（md={min_duration}）: {existing[-1].name}", flush=True)
        return existing[-1]
    run_cli([
        "-m", "benchmark.cli.eval",
        "--config", str(variant),
        "--ckpt", str(ckpt),
        "--out-dir", str(out_dir),
    ])
    evals = sorted(out_dir.glob("*.evaluation.json"))
    if not evals:
        raise SystemExit(f"{run_dir} 口径变体评估未产出 evaluation.json")
    return evals[-1]


def recall_ceiling(eval_path: Path, data: dict) -> tuple[dict[str, float], int]:
    """按真值段长算逐类召回上限：段长 ≥ min_duration 的帧占比。

    因果平滑的 ``min_duration`` 是硬上限——短于它的真实段在输出里不可能出现。
    阈值从评估结果的 ``inference.smoothing`` 字符串解析（形如
    ``causal_decision(min_duration=25)``），口径随实际评估配置走，不写死。
    """

    smoothing = str((data.get("inference") or {}).get("smoothing") or "")
    match = re.search(r"min_duration=(\d+)", smoothing)
    min_duration = int(match.group(1)) if match else 0

    run_dir = eval_path.parent.parent
    artifact = run_dir / ((data.get("artifacts") or {}).get("predictions", {}).get("path") or "")
    if not artifact.is_file() or min_duration <= 0:
        return {}, min_duration

    lengths: dict[str, list[int]] = {}
    for item in json.loads(artifact.read_text(encoding="utf-8"))["items"].values():
        truth = item["truth_labels"]
        start = 0
        for index in range(1, len(truth) + 1):
            if index == len(truth) or truth[index] != truth[start]:
                lengths.setdefault(truth[start], []).append(index - start)
                start = index
    ceiling = {}
    for name, values in lengths.items():
        total = sum(values)
        if total:
            ceiling[name] = sum(length for length in values if length >= min_duration) / total
    return ceiling, min_duration


def linear_probe_row(run_dir: Path, min_duration: int | None = None) -> dict:
    """逐帧线性探针参照行：train 拟合等先验多类 LDA，按**与模型相同的推理口径**评估。

    这是"特征里有多少可用信号 + 相同后处理"的端到端下界：LDA 判别分数当作 logits，
    用同一个 `causal_decision(min_duration=…)` 与同样的冷启动填充（前 window-1 帧 idle），
    指标走同一套 `temporal_metrics`。因此它可以直接与模型行比较——模型若不明显高于它，
    说明换特征/加时序没有带来增益。LDA 为闭式解（纯 numpy，与
    `tools/probe_split_shift.py` 同源），不训练、不引入依赖。
    """

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根：import framework
    sys.path.insert(0, str(Path(__file__).resolve().parent))     # tools/：import probe_split_shift
    import torch

    from framework.cleansight_eval.core.config import load_config
    from framework.cleansight_eval.core.metrics import temporal_metrics
    from framework.cleansight_eval.temporal.data import load_split, split_video_names
    from framework.cleansight_eval.temporal.util import causal_decision

    from probe_split_shift import fit_multiclass_lda

    cfg = load_config(run_dir / "config.resolved.json")
    data, feature_schema = cfg["data"], cfg["feature_schema"]
    n_classes = int(cfg["model"]["num_classes"])
    split = data["split_eval"]
    # 探针模拟"同一条推理协议"：滑窗因果模型有 window 冷启动 + min_duration 平滑；
    # 全序列模型（MS-TCN / Transformer）两者都没有，探针也不该套用（否则人为压低参照）。
    causal = str(cfg.get("pipeline") or "") == "sliding_window_temporal"
    window = int(cfg["train"].get("window", 64)) if causal else 1
    if min_duration is None:
        min_duration = int((cfg.get("evaluation") or {}).get("smoothing_min_duration", 25))
    if not causal:
        min_duration = 1

    train_feats, train_truths, id2name = load_split(data, data["split_train"],
                                                    feature_schema=feature_schema)
    x_train = np.concatenate(train_feats).astype(np.float64)
    y_train = np.concatenate(train_truths)
    w, b = fit_multiclass_lda(x_train, y_train, n_classes)

    feats, truths, _ = load_split(data, split, feature_schema=feature_schema)
    names = split_video_names(data, split)
    labels = [id2name[i] for i in range(n_classes)]
    pred_by_item, truth_by_item = {}, {}
    for name, frames, truth in zip(names, feats, truths):
        scores = frames.astype(np.float64) @ w.T + b  # [T, C]，当 logits 用
        predicted = np.zeros(len(scores), dtype=np.int64)
        predicted[: window - 1] = 0  # 与滑窗模型一致的冷启动填充
        pending, stable, count = None, 0, 0
        for index in range(window - 1, len(scores)):
            pending, stable, count = causal_decision(
                torch.from_numpy(scores[index]).float(), pending, stable, count,
                min_duration=min_duration,
            )
            predicted[index] = stable
        pred_by_item[name] = [labels[int(v)] for v in predicted]
        truth_by_item[name] = [labels[int(v)] for v in truth]

    raw = temporal_metrics(pred_by_item, truth_by_item, labels=labels)
    frame, segment = raw["frame"], raw["segment"]
    per_class = frame.get("per_class") or {}
    conf = frame["confusion_matrix_rows_truth_cols_prediction"]
    return {
        "strategy": f"linear-probe（逐帧 LDA），md={min_duration}",
        "seed": 0,
        "acc": frame["accuracy"] * 100,
        "edit": segment["edit"] * 100,
        "f1_01": segment["f1_at_iou"]["0.10"] * 100,
        "f1_025": segment["f1_at_iou"]["0.25"] * 100,
        "nonidle": sum(sum(conf[r][c] for r in range(len(conf))) for c in range(1, len(conf[0]))),
        "per_class_recall": {name: (per_class.get(name) or {}).get("recall") for name in labels},
        "per_class_support": {name: (per_class.get(name) or {}).get("support") for name in labels},
        "recall_ceiling": {},
        "smoothing_min_duration": min_duration,
        "probe": True,
    }


def summarize(runs_dir: Path, rows: list[dict]) -> str:
    """按策略聚合逐 seed 指标与中位数，返回 markdown 摘要文本。

    testset 身份（id / revision / 样本数）与逐类 support 均从各 run 的 evaluation.json 读取，
    不在这里写死数据口径——数据换版（如 2026-09-17 起 test 改为 project-18 的 8 视频）后
    摘要自动跟随。逐类 recall 取各 seed 的中位数，support=0 的类标 n/a（该 split 不可评估）。
    """

    testsets = {(row.get("testset_split"), row.get("testset_revision"), row.get("testset_items"))
                for row in rows if not row.get("probe")}  # 探针行不参与口径分组
    if len(testsets) == 1:
        ts_split, ts_rev, ts_items = testsets.pop()
        data_line = (f"- 数据：split `{ts_split}`，revision `{(ts_rev or '?')[:8]}…`，"
                     f"{ts_items} 个样本视频（各策略契约共用同一 manifest）")
    else:
        data_line = (f"- 数据：本矩阵含 {len(testsets)} 个不同 testset 口径，"
                     f"逐 run 见各自 evaluation.json")
    model_rows = [row for row in rows if not row.get("probe")]
    smoothing = (model_rows[0].get("smoothing_min_duration") if model_rows else None)
    lines = [
        "# 特征策略对照矩阵汇总（一键复跑）",
        "",
        f"- 配方：weight_decay={DEFAULT_WD} / dropout={DEFAULT_DROPOUT} / "
        f"patience={DEFAULT_PATIENCE} / best_metric={DEFAULT_BEST_METRIC}",
        f"- 评估口径：causal_decision(min_duration={smoothing})"
        + ("（历史默认 25）" if smoothing == 25 else "（非默认，见 docs/FEATURE_STRATEGY_COMPARE.md 第七/八轮）"),
        data_line,
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
        if len(items) == 1 and items[0].get("probe"):
            continue  # 探针参照只有一行，不需要中位数
        lines.append(
            f"| **{strategy} 中位数** | — | {median('acc'):.2f} | {median('edit'):.2f} "
            f"| {median('f1_01'):.2f} | {median('f1_025'):.2f} | — |"
        )

    labels = (rows[0].get("labels") if rows else None) or []
    if labels:
        support = rows[0].get("per_class_support") or {}

        def cell(row: dict, label: str) -> str:
            value = (row.get("per_class_recall") or {}).get(label)
            return "n/a" if value is None else f"{value * 100:.1f}"

        lines += [
            "",
            "## 逐类帧级 recall（% ，各 seed 中位数；support 为该类 truth 帧数）",
            "",
            "> **召回上限**：因果平滑 `smoothing_min_duration` 是硬上限——真实段长短于该值的类",
            "> 不可能被输出（下表末行为按真值段长算的上限，用于区分「协议压制」与「模型没学到」）。",
            "> `linear-probe（…）` 行为逐帧 LDA 参照：train 拟合、**与模型同一个 min_duration 与冷启动**，",
            "> 指标走同一套 `temporal_metrics`——模型不明显高于它，说明换特征/加时序没有增益。",
            "",
            "| 策略 | " + " | ".join(labels) + " |",
            "|---" * (len(labels) + 1) + "|",
            "| **support（truth 帧）** | "
            + " | ".join(str(support.get(label, "?")) for label in labels) + " |",
        ]
        ceiling = rows[0].get("recall_ceiling") or {}
        min_duration = rows[0].get("smoothing_min_duration")
        if ceiling and labels:
            # 默认类（id 0 = idle）是平滑的初始 stable 态，不受最小持续时长约束，故不列上限
            default_label = labels[0]
            lines.append(
                f"| **召回上限（非 {default_label} 类，段长≥{min_duration}）** | "
                + " | ".join(
                    "—" if label == default_label or label not in ceiling
                    else f"{ceiling[label] * 100:.1f}" for label in labels
                ) + " |"
            )
        for strategy in sorted(per_strategy):
            items = sorted(per_strategy[strategy], key=lambda r: r["seed"])
            cells = []
            for label in labels:
                values = [row["per_class_recall"].get(label) for row in items]
                values = [v for v in values if v is not None]
                cells.append("n/a" if not values else f"{statistics.median(values) * 100:.1f}")
            lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
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
    parser.add_argument("--no-probe-baseline", action="store_true",
                        help="不计算逐帧线性探针参照行（默认计算，用于给出特征层下界）")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="extra_sets",
                        help="附加 -S 覆盖（可多次），如 --set model.normalization=zscore")
    parser.add_argument("--smoothing-min-duration", type=int, default=None,
                        help="评估口径：覆盖 evaluation.smoothing_min_duration（默认沿用配置）")
    parser.add_argument("--force-eval", action="store_true",
                        help="强制重新评估（默认复用 evals/ 下已有产物，汇总幂等）")
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
    for item in args.extra_sets:
        recipe.extend(["-S", item])

    run_dirs: list[tuple[str, int, Path]] = []
    if not args.skip_train:
        for strategy in strategies:
            for seed in seeds:
                cfg = STRATEGIES[strategy]
                print(f"== 训练 {strategy} seed={seed} ==", flush=True)
                run_dir = Path(train_config(cfg, seed, runs_dir, recipe))
                run_dirs.append((strategy, seed, run_dir))
    else:
        seen: dict[tuple[str, int], Path] = {}
        # 不能只 glob "gru-*"：MS-TCN / Transformer 的 run 目录以各自 model.type 命名。
        for run_dir in sorted(path for path in runs_dir.iterdir()
                              if path.is_dir() and (path / "config.resolved.json").is_file()):
            key = (strategy_name_of_run(run_dir), seed_of_run(run_dir))
            if key in seen:
                print(f"== 跳过重复 run {run_dir.name}（{key[0]} seed={key[1]} 已有 {seen[key].name}）==",
                      flush=True)
            seen[key] = run_dir  # 目录名按时间戳排序，保留最新
        for (strategy, seed), run_dir in sorted(seen.items()):
            run_dirs.append((strategy, seed, run_dir))

    rows: list[dict] = []
    probe_rows: list[dict] = []
    if not args.skip_eval:
        for strategy, seed, run_dir in run_dirs:
            print(f"== 评估 {run_dir.name} ==", flush=True)
            eval_path = evaluate_run(run_dir, reuse=not args.force_eval,
                                      min_duration=args.smoothing_min_duration)
            data = json.loads(eval_path.read_text(encoding="utf-8"))
            summary = data["metrics"]["summary"]
            temporal = data["metrics"]["details"]["temporal"]
            seg = temporal["segment"]
            cm = temporal["frame"]["confusion_matrix_rows_truth_cols_prediction"]
            per_class = temporal["frame"].get("per_class") or {}
            testset = data.get("testset") or {}
            ceiling, min_duration = recall_ceiling(eval_path, data)
            rows.append({
                "strategy": strategy if strategy != "?" else Path(config_file_of_run(run_dir)).stem,
                "seed": seed,
                "acc": summary["acc"]["value"],
                "edit": summary["edit"]["value"],
                "f1_01": seg["f1_at_iou"]["0.10"] * 100,
                "f1_025": seg["f1_at_iou"]["0.25"] * 100,
                "nonidle": sum(sum(cm[r][c] for r in range(len(cm))) for c in range(1, len(cm[0]))),
                # testset 身份与逐类指标：供摘要自证口径，不写死数据版本
                "testset_id": testset.get("id"),
                "testset_split": testset.get("split"),
                "testset_revision": testset.get("dataset_revision"),
                "testset_items": testset.get("num_items"),
                "labels": testset.get("labels") or [],
                "per_class_recall": {k: (v or {}).get("recall") for k, v in per_class.items()},
                "recall_ceiling": ceiling,
                "smoothing_min_duration": min_duration,
                "per_class_support": {k: (v or {}).get("support") for k, v in per_class.items()},
            })

    if not args.skip_eval and not args.no_probe_baseline and run_dirs:
        seen_probe: set[tuple] = set()
        for _strategy, _seed, run_dir in run_dirs:
            try:
                key = (strategy_name_of_run(run_dir),)
                if key in seen_probe:
                    continue
                seen_probe.add(key)
                print(f"== 逐帧线性探针参照（{key[0]}，无时序/无平滑）==", flush=True)
                probe = linear_probe_row(run_dir, args.smoothing_min_duration)
                probe["strategy"] = f"linear-probe（{key[0]}），md={probe['smoothing_min_duration'] or 25}"
                probe_rows.append(probe)
            except Exception as exc:  # 探针失败不应让汇总整体失败
                print(f"  探针参照计算失败（跳过）: {exc}", flush=True)
    rows.extend(probe_rows)

    text = summarize(runs_dir, rows)
    print(text)
    (runs_dir / "STRATEGY_SUMMARY.md").write_text(text, encoding="utf-8")
    print(f"摘要已写入 {runs_dir / 'STRATEGY_SUMMARY.md'}")


if __name__ == "__main__":
    sys.exit(main())
