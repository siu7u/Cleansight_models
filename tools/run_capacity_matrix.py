"""模型容量（参数量）对照矩阵：python tools/run_capacity_matrix.py。

回答「不同规模参数量的模型对**训练**与**预测内容**有什么影响」：同一份实验配置、同一份
数据、同一 seed 集合，只变 ``model.hidden``（参数量）——每个容量点训练若干个 seed，
对 test split 做正式评估，最后汇总成一张可复跑的 markdown 表。

三类读数（都取自 run 既有产物，不新增口径）：

- **训练侧**：``history.csv``（首/末轮 loss、每轮秒数）+ ``status.json``（best epoch）；
  固定 ``train.epochs`` 且默认**关闭早停**，否则容量大的模型更容易被 val_loss 提前截断，
  容量差异与"训练是否跑满"会纠缠在一起。
- **预测内容侧**：``evaluation.json``——段级 edit / F1@0.1,0.25,0.5、``tp/fp/fn@0.5``
  （``fp`` 是过分割的绝对量，``预测段数/真值段数``是相对量）、逐类帧级 recall（含 support）、
  非 idle 预测帧数（坍缩解自查）。
- **噪声地板**：同一容量点跨 seed 的中位数与摆幅（max−min）并排给出。这个数据集只有
  14/4/8 个视频（train/val/test），容量差异小于摆幅时不可判读为"容量效应"。

``best.pt`` 与 ``last.pt`` 都评估（``--no-eval-last`` 可关）：val 只有 4 个视频，
``best_metric=val_f1_0.5`` 的选点本身带噪声，固定预算下的 last.pt 是不经选点的对照。
last 的评估产物写在 ``<runs-dir>/_eval_last/<run>/``，不覆盖 run 自己的 ``evals/``。

训练/评估/探针原语复用 ``tools/run_strategy_matrix.py``（同一套 ``benchmark.cli.eval``
调用、同一套 reuse 语义、同一份逐帧线性探针参照），这里只加"容量"这一维与它的汇总表。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# 脚本入口（python tools/run_capacity_matrix.py）下 tools/ 已在 sys.path；显式注入以支持
# 从其它 cwd 调用。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_strategy_matrix import (  # noqa: E402  路径需先注入
    DEFAULT_BEST_METRIC,
    DEFAULT_WD,
    eval_config_of_run,
    evaluate_run,
    linear_probe_row,
    run_cli,
    train_config,
)

SUMMARY_NAME = "CAPACITY_SUMMARY.md"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="模型容量（参数量）× 多 seed 对照矩阵")
    p.add_argument("--runs-dir", default="runs/capacity-mstcn", help="批次输出根目录（每容量点一个子目录）")
    p.add_argument("--config", default="mstcn-actionmixed-auto-roi.yaml",
                   help="framework/experiments 下的配置文件名（容量点共用它的数据/特征契约）")
    p.add_argument("--hidden", default="16,32,64,128,256", help="容量点（model.hidden，逗号分隔）")
    p.add_argument("--seeds", default="42,7,2026", help="seed 列表（逗号分隔）")
    p.add_argument("--epochs", type=int, default=30, help="固定训练预算（容量对照必须同预算）")
    p.add_argument("--patience", type=int, default=None,
                   help="早停轮数；缺省关闭（容量对照下早停会与截断混淆）")
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WD)
    p.add_argument("--best-metric", default=DEFAULT_BEST_METRIC)
    p.add_argument("--skip-train", action="store_true", help="只评估+汇总已有 run")
    p.add_argument("--skip-eval", action="store_true", help="只训练+汇总（无 test 指标）")
    p.add_argument("--no-eval-last", action="store_true", help="不评估 last.pt（默认评估，用于对照 val 选点噪声）")
    p.add_argument("--no-probe-baseline", action="store_true", help="不计算逐帧线性探针参照行")
    p.add_argument("--force-eval", action="store_true", help="忽略已有评估产物，重跑评估")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="extra_sets",
                   help="附加 -S 覆盖（可多次），如 --set model.dropout=0.1")
    return p.parse_args(argv)


def build_recipe(args) -> list[str]:
    """构造训练 -S 覆盖：容量点之外的配方在全部点上保持一致。

    ``train.patience=null`` 经 ``cli/train.py`` 的 ``_coerce`` 解析为 None，即关闭早停。
    """

    patience = "null" if args.patience is None else str(args.patience)
    recipe = [
        "-S", f"train.weight_decay={args.weight_decay}",
        "-S", f"train.epochs={args.epochs}",
        "-S", f"train.best_metric={args.best_metric}",
        "-S", f"train.patience={patience}",
    ]
    for item in args.extra_sets:
        recipe.extend(["-S", item])
    return recipe


def evaluate_last_ckpt(run_dir: Path, *, reuse: bool = True) -> Path:
    """评估 ``checkpoints/last.pt``（固定预算、不经 val 选点），产物落 ``_eval_last/<run>/``。"""

    ckpt = run_dir / "checkpoints" / "last.pt"
    if not ckpt.is_file():
        raise SystemExit(f"{run_dir} 缺少 last.pt")
    out_dir = run_dir.parent / "_eval_last" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.evaluation.json"))
    if reuse and existing:
        print(f"  复用已有 last 评估: {existing[-1].name}", flush=True)
        return existing[-1]
    run_cli([
        "-m", "benchmark.cli.eval",
        "--config", str(eval_config_of_run(run_dir)),
        "--ckpt", str(ckpt),
        "--out-dir", str(out_dir),
    ])
    evals = sorted(out_dir.glob("*.evaluation.json"))
    if not evals:
        raise SystemExit(f"{run_dir} last.pt 评估未产出 evaluation.json")
    return evals[-1]


def read_history(run_dir: Path) -> dict:
    """读 ``history.csv``：首/末轮 loss、最低 val_loss、每轮秒数中位数。"""

    with (run_dir / "history.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def column(name: str) -> list[float]:
        values = []
        for row in rows:
            raw = row.get(name)
            if raw not in (None, ""):
                values.append(float(raw))
        return values

    train_loss, val_loss, secs = column("train_loss"), column("val_loss"), column("epoch_sec")
    return {
        "history_epochs": len(rows),
        "first_train_loss": train_loss[0] if train_loss else None,
        "last_train_loss": train_loss[-1] if train_loss else None,
        "last_val_loss": val_loss[-1] if val_loss else None,
        "min_val_loss": min(val_loss) if val_loss else None,
        "epoch_sec_median": statistics.median(secs) if secs else None,
    }


def read_best_epoch(run_dir: Path) -> int | None:
    """从 ``status.json`` 读 best_metric.epoch（best.pt 落在第几轮）。"""

    path = run_dir / "status.json"
    if not path.is_file():
        return None
    best = (json.loads(path.read_text(encoding="utf-8")) or {}).get("best_metric") or {}
    epoch = best.get("epoch")
    return int(epoch) if isinstance(epoch, int) else None


def read_params(run_dir: Path) -> int | None:
    """从 checkpoint meta 读 num_params（与评估产物里的 ``model.num_params`` 应一致）。"""

    meta = run_dir / "checkpoints" / "best.pt.meta.json"
    if not meta.is_file():
        return None
    value = (json.loads(meta.read_text(encoding="utf-8")) or {}).get("num_params")
    return int(value) if isinstance(value, int) else None


def read_metrics(eval_path: Path) -> dict:
    """从 evaluation.json 摘出预测内容侧读数（帧级 + 段级 + 逐类 + 过分割计数）。"""

    data = json.loads(eval_path.read_text(encoding="utf-8"))
    temporal = data["metrics"]["details"]["temporal"]
    frame, segment = temporal["frame"], temporal["segment"]
    detail = segment["details_at_iou"]["0.50"]
    confusion = frame["confusion_matrix_rows_truth_cols_prediction"]
    per_class = frame.get("per_class") or {}
    tp, fp, fn = detail["tp"], detail["fp"], detail["fn"]
    testset = data.get("testset") or {}
    return {
        "acc": frame["accuracy"] * 100,
        "edit": segment["edit"] * 100,
        "f1_01": segment["f1_at_iou"]["0.10"] * 100,
        "f1_025": segment["f1_at_iou"]["0.25"] * 100,
        "f1_05": segment["f1_at_iou"]["0.50"] * 100,
        "tp": tp, "fp": fp, "fn": fn,
        "seg_pred": tp + fp,
        "seg_true": tp + fn,
        "seg_ratio": (tp + fp) / (tp + fn) if (tp + fn) else None,
        "precision_05": detail["precision"] * 100,
        "recall_05": detail["recall"] * 100,
        "nonidle": sum(sum(confusion[r][c] for r in range(len(confusion)))
                       for c in range(1, len(confusion[0]))),
        "per_class_recall": {k: (v or {}).get("recall") for k, v in per_class.items()},
        "per_class_support": {k: (v or {}).get("support") for k, v in per_class.items()},
        "labels": testset.get("labels") or [],
        "num_params": (data.get("model") or {}).get("num_params"),
        "testset": testset,
    }


def discover_runs(runs_dir: Path, config_name: str) -> list[tuple[int, int, Path]]:
    """``--skip-train`` 用：扫描已有 run，按 (hidden, seed) 去重并保留最新一个。

    hidden 取 run 的 ``config.resolved.json``（= 实际生效值），seed 取 ``env.json``；
    与源配置文件名不符的 run 一律跳过，避免把别的批次混进容量对照。
    """

    found: dict[tuple[int, int], Path] = {}
    for run_dir in sorted(path for path in runs_dir.glob("*/mstcn-*")
                          if (path / "config.resolved.json").is_file()):
        cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
        source = ((cfg.get("_config_provenance") or {}).get("source_path") or "")
        if Path(source).name != config_name:
            continue
        hidden = (cfg.get("model") or {}).get("hidden")
        seed = (json.loads((run_dir / "env.json").read_text(encoding="utf-8")) or {}).get("seed")
        if not isinstance(hidden, int) or not isinstance(seed, int):
            continue
        found[(hidden, seed)] = run_dir  # 目录名按时间戳排序，保留最新
    return [(hidden, seed, run_dir) for (hidden, seed), run_dir in sorted(found.items())]


def median(values: list[float]) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def swing(values: list[float]) -> float | None:
    """跨 seed 摆幅（max−min）：同容量点的噪声地板，结论必须大于它。"""

    values = [value for value in values if value is not None]
    return max(values) - min(values) if len(values) > 1 else None


def fmt(value, digits: int = 2, dash: str = "—") -> str:
    if value is None:
        return dash
    return f"{value:.{digits}f}"


def render_summary(args, rows: list[dict], probe: dict | None, runs_dir: Path) -> str:
    """渲染 markdown 汇总：主表（best/last）+ 训练侧 + 噪声地板 + 逐类 recall。"""

    recipe = (f"- 配方：config=`framework/experiments/{args.config}` / seeds="
              f"{','.join(str(s) for s in sorted({row['seed'] for row in rows}))} / "
              f"epochs={args.epochs} / patience={args.patience if args.patience else '关闭'} / "
              f"weight_decay={args.weight_decay} / best_metric={args.best_metric}"
              + (f" / 附加 --set {' '.join(args.extra_sets)}" if args.extra_sets else ""))
    testsets = {(row["test"]["testset"].get("split"), row["test"]["testset"].get("dataset_revision"),
                 row["test"]["testset"].get("num_items")) for row in rows if row.get("test")}
    if len(testsets) == 1:
        split, revision, items = testsets.pop()
        data_line = (f"- 数据：split `{split}`，revision `{str(revision or '?')[:8]}…`，"
                     f"{items} 个样本视频（各容量点共用同一 manifest）")
    else:
        data_line = f"- 数据：本矩阵含 {len(testsets)} 个不同 testset 口径，逐 run 见各自 evaluation.json"

    lines = [
        "# 模型容量（参数量）对照矩阵汇总（一键复跑）",
        "",
        recipe,
        data_line,
        f"- 评估口径：`benchmark.cli.eval` 正式评估（全序列模型无因果平滑，md 不适用）",
        f"- 运行目录：`{runs_dir}`",
        "- 判读规则：容量点之间的差异必须大于同容量点的**跨 seed 摆幅**（噪声地板）才可判读为容量效应；",
        "  `best.pt` 由 val（4 个视频）的 `val_f1_0.5` 选出，本身带选点噪声，故同时给出 `last.pt` 对照。",
        "",
        "## 1. test 指标（best.pt，按 val 选点）",
        "",
        "| 容量 | 参数量 | seed | acc | edit | F1@0.1 | F1@0.25 | F1@0.5 | tp/fp/fn@0.5 | 段数比(预测/真值) | 非idle帧 | best epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    by_capacity: dict[int, list[dict]] = {}
    for row in rows:
        by_capacity.setdefault(row["hidden"], []).append(row)

    def metric_row(row: dict, metrics: dict | None, label: str, params: str, extra: str) -> str:
        if metrics is None:
            return f"| {label} | {params} | {row['seed']} | — | — | — | — | — | — | — | — | {extra} |"
        return (f"| {label} | {params} | {row['seed']} | {fmt(metrics['acc'])} | {fmt(metrics['edit'])} "
                f"| {fmt(metrics['f1_01'])} | {fmt(metrics['f1_025'])} | {fmt(metrics['f1_05'])} "
                f"| {metrics['tp']}/{metrics['fp']}/{metrics['fn']} | {fmt(metrics['seg_ratio'])} "
                f"| {metrics['nonidle']} | {extra} |")

    for hidden in sorted(by_capacity):
        items = sorted(by_capacity[hidden], key=lambda row: row["seed"])
        params = items[0]["params"]
        params_text = f"{params:,}" if isinstance(params, int) else "?"
        for row in items:
            lines.append(metric_row(row, row.get("test"), f"h{hidden}", params_text,
                                    str(row.get("best_epoch") or "—")))
        values = [row["test"] for row in items if row.get("test")]
        if values:
            lines.append(
                f"| **h{hidden} 中位数** | {params_text} | — | {fmt(median([v['acc'] for v in values]))} "
                f"| {fmt(median([v['edit'] for v in values]))} | {fmt(median([v['f1_01'] for v in values]))} "
                f"| {fmt(median([v['f1_025'] for v in values]))} | {fmt(median([v['f1_05'] for v in values]))} "
                f"| — | {fmt(median([v['seg_ratio'] for v in values]))} "
                f"| {fmt(median([float(v['nonidle']) for v in values]), 0)} | — |"
            )

    lines += [
        "",
        "## 2. test 指标（last.pt，固定预算不经选点）",
        "",
        "| 容量 | 参数量 | seed | acc | edit | F1@0.1 | F1@0.25 | F1@0.5 | tp/fp/fn@0.5 | 段数比(预测/真值) | 非idle帧 | 总轮数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for hidden in sorted(by_capacity):
        items = sorted(by_capacity[hidden], key=lambda row: row["seed"])
        params = items[0]["params"]
        params_text = f"{params:,}" if isinstance(params, int) else "?"
        for row in items:
            lines.append(metric_row(row, row.get("test_last"), f"h{hidden}", params_text,
                                    str(row["history"].get("history_epochs") or "—")))
        values = [row["test_last"] for row in items if row.get("test_last")]
        if values:
            lines.append(
                f"| **h{hidden} 中位数** | {params_text} | — | {fmt(median([v['acc'] for v in values]))} "
                f"| {fmt(median([v['edit'] for v in values]))} | {fmt(median([v['f1_01'] for v in values]))} "
                f"| {fmt(median([v['f1_025'] for v in values]))} | {fmt(median([v['f1_05'] for v in values]))} "
                f"| — | {fmt(median([v['seg_ratio'] for v in values]))} "
                f"| {fmt(median([float(v['nonidle']) for v in values]), 0)} | — |"
            )

    lines += [
        "",
        "## 3. 训练侧（history.csv / status.json）",
        "",
        "| 容量 | seed | 首轮 train_loss | 末轮 train_loss | 末轮 val_loss | 最低 val_loss | best epoch | 每轮秒（中位） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hidden in sorted(by_capacity):
        items = sorted(by_capacity[hidden], key=lambda row: row["seed"])
        for row in items:
            history = row["history"]
            lines.append(
                f"| h{hidden} | {row['seed']} | {fmt(history['first_train_loss'], 3)} "
                f"| {fmt(history['last_train_loss'], 3)} | {fmt(history['last_val_loss'], 3)} "
                f"| {fmt(history['min_val_loss'], 3)} | {row.get('best_epoch') or '—'} "
                f"| {fmt(history['epoch_sec_median'], 3)} |"
            )
        lines.append(
            f"| **h{hidden} 中位数** | — | {fmt(median([r['history']['first_train_loss'] for r in items]), 3)} "
            f"| {fmt(median([r['history']['last_train_loss'] for r in items]), 3)} "
            f"| {fmt(median([r['history']['last_val_loss'] for r in items]), 3)} "
            f"| {fmt(median([r['history']['min_val_loss'] for r in items]), 3)} | — "
            f"| {fmt(median([r['history']['epoch_sec_median'] for r in items]), 3)} |"
        )

    lines += [
        "",
        "## 4. 噪声地板（同容量点跨 seed 摆幅）与容量读数",
        "",
        "> 摆幅 = 同容量点跨 seed 的 max−min，是**该点的噪声地板**：容量点之间的中位数差小于它就不",
        "> 可判读为容量效应。`acc 下界（全 idle）` = test 里 idle 帧占比（多数类基线）；正式评估的",
        "> acc 若与之齐平，说明模型退化为「全 idle」，段级指标才是有效读数。",
        "",
        "| 容量 | 参数量 | acc 中位 | acc 摆幅 | edit 中位 | edit 摆幅 | F1@0.25 中位 | F1@0.25 摆幅 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hidden in sorted(by_capacity):
        items = [row["test"] for row in by_capacity[hidden] if row.get("test")]
        params = by_capacity[hidden][0]["params"]
        params_text = f"{params:,}" if isinstance(params, int) else "?"
        if not items:
            lines.append(f"| h{hidden} | {params_text} | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| h{hidden} | {params_text} | {fmt(median([v['acc'] for v in items]))} "
            f"| {fmt(swing([v['acc'] for v in items]))} | {fmt(median([v['edit'] for v in items]))} "
            f"| {fmt(swing([v['edit'] for v in items]))} | {fmt(median([v['f1_025'] for v in items]))} "
            f"| {fmt(swing([v['f1_025'] for v in items]))} |"
        )
    if probe:
        lines.append(
            f"| {probe['strategy']} | — | {fmt(probe['acc'])} | — | {fmt(probe['edit'])} | — "
            f"| {fmt(probe['f1_025'])} | — |"
        )
    first_test = next((row["test"] for row in rows if row.get("test")), None)
    if first_test:
        support = first_test["per_class_support"]
        total = sum(value for value in support.values() if isinstance(value, int))
        idle_support = support.get((first_test["labels"] or ["idle"])[0])
        if total and isinstance(idle_support, int):
            lines.append(
                f"| acc 下界（全 idle，多数类） | — | {fmt(idle_support / total * 100)} | — | — | — | — | — |"
            )

    labels = next((row["test"]["labels"] for row in rows if row.get("test")), []) or []
    if labels:
        support = next((row["test"]["per_class_support"] for row in rows if row.get("test")), {})
        lines += [
            "",
            "## 5. 逐类帧级 recall（%，各 seed 中位数；support 为该类 truth 帧数）",
            "",
            "| 容量 | " + " | ".join(labels) + " |",
            "|---" * (len(labels) + 1) + "|",
            "| **support（truth 帧）** | "
            + " | ".join(str(support.get(label, "?")) for label in labels) + " |",
        ]
        if probe:
            lines.append("| " + probe["strategy"] + " | " + " | ".join(
                "n/a" if probe["per_class_recall"].get(label) is None
                else f"{probe['per_class_recall'][label] * 100:.1f}" for label in labels) + " |")
        for hidden in sorted(by_capacity):
            items = [row["test"] for row in by_capacity[hidden] if row.get("test")]
            cells = []
            for label in labels:
                values = [v["per_class_recall"].get(label) for v in items]
                values = [value for value in values if value is not None]
                cells.append("n/a" if not values else f"{median(values) * 100:.1f}")
            lines.append(f"| h{hidden} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "> support=0 的类在该 split 上不可评估（n/a）。线性探针行 = 逐帧 LDA（train 拟合、",
            "> 与全序列模型同为无平滑无冷启动），是「特征里有多少可用信号」的端到端下界：",
            "> 容量最大的点若不明显高于它，说明瓶颈在特征/数据而非参数量。",
        ]
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    runs_dir = (REPO_ROOT / args.runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    hidden_points = [int(value) for value in args.hidden.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    recipe = build_recipe(args)

    run_dirs: list[tuple[int, int, Path]] = []
    if args.skip_train:
        run_dirs = discover_runs(runs_dir, args.config)
        print(f"[capacity] 复用已有 run {len(run_dirs)} 个", flush=True)
    else:
        for hidden in hidden_points:
            for seed in seeds:
                print(f"== 训练 hidden={hidden} seed={seed} ==", flush=True)
                run_dir = Path(train_config(
                    args.config, seed, runs_dir / f"h{hidden}",
                    recipe + ["-S", f"model.hidden={hidden}"],
                ))
                run_dirs.append((hidden, seed, run_dir))

    rows: list[dict] = []
    for hidden, seed, run_dir in run_dirs:
        row = {
            "hidden": hidden,
            "seed": seed,
            "run_dir": run_dir,
            "params": read_params(run_dir),
            "best_epoch": read_best_epoch(run_dir),
            "history": read_history(run_dir),
            "test": None,
            "test_last": None,
        }
        if not args.skip_eval:
            print(f"== 评估 {run_dir.name}（best.pt）==", flush=True)
            row["test"] = read_metrics(evaluate_run(run_dir, reuse=not args.force_eval))
            if not args.no_eval_last:
                print(f"== 评估 {run_dir.name}（last.pt）==", flush=True)
                row["test_last"] = read_metrics(evaluate_last_ckpt(run_dir, reuse=not args.force_eval))
        rows.append(row)

    probe = None
    if not args.no_probe_baseline and rows:
        try:
            print("== 逐帧线性探针参照（容量无关下界）==", flush=True)
            probe = linear_probe_row(rows[0]["run_dir"], None)
            probe["strategy"] = probe["strategy"].replace("（逐帧 LDA）", "（逐帧 LDA，容量无关下界）")
        except Exception as exc:  # 探针失败不应让汇总整体失败
            print(f"  探针参照计算失败（跳过）: {exc}", flush=True)

    text = render_summary(args, rows, probe, runs_dir)
    print(text)
    (runs_dir / SUMMARY_NAME).write_text(text, encoding="utf-8")
    print(f"摘要已写入 {runs_dir / SUMMARY_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
