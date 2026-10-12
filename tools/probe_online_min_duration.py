"""在线轨 `smoothing_min_duration` 扫描探针：同一份 checkpoint、只改这一个门槛，逐 seed 配对比较。

## 为什么需要它

在线轨（`sliding_window_temporal`）的推理后处理 `causal_decision(..., min_duration=25)` 里，
`min_duration` 是**硬召回上限**——候选类必须连续出现 ≥ 该帧数才会被写成 `stable`，故真实段短于
该值的动作**在输出里不可能出现**。而扫过的 6 个在线臂 `evaluation.smoothing_min_duration`
**全部为 None**（走默认 25），这个常数从未被扫过（见
`docs/weeks/2026-10-10_2026-10-16/ONLINE_MIN_DURATION_20261012.md`）。

它**无法离线重套**：在线产物只存迟滞之后的标签（`predicted_label_ids`），
而门槛的输入是**未过门槛的候选序列**，该序列没被存下来。所以只能重跑推理——
本工具就是把"重跑推理"这件事做成可复现的一步。

## 三个必须做对的地方（前几轮踩过）

1. **口径**：`benchmark.cli.eval` 的参数只有 `--config/--ckpt/--out-dir`，
   **没有 `-S` 覆盖、也没有 `--split`**。故本工具复制 `config.resolved.json` 到临时目录、
   就地改写 `evaluation.smoothing_min_duration` 与 `data.split_eval`。
2. **不污染原 run**：评估器默认把报告写到 ckpt 所在 run（2026-10-11 事故成因），
   故这里 ① 把 `best.pt`(+`.meta.json`) **硬链接**到临时 run；② 显式传 `--out-dir`。
3. **选参纪律**：默认在 **val** 上扫（`--split val`）；要报 test 请显式 `--split test`，
   并确保门槛是在 val 上选出来的。

用法（仓库根执行）::

    # 先在 val 上给一个臂扫门槛（默认 --split val）
    python tools/probe_online_min_duration.py \
        --arm "runs/v41-push/online-gru-hand-v2/h128/gru-*" --md 25 10

    # 全部 seed 配对后再去 test 报数
    python tools/probe_online_min_duration.py --arm "<glob>" --md 25 10 --split test
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "tmp" / "online-md-probe"


def load_arm(run_dir: Path) -> dict:
    """读一个在线臂 run 目录的配置与 seed。"""

    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    return {"dir": run_dir, "cfg": cfg, "seed": env["seed"]}


def prepare(run_dir: Path, md: int, split: str) -> tuple[Path, Path]:
    """搭一个隔离的临时 run：硬链接 checkpoint + 改写后的配置。返回 (config, ckpt)。"""

    work = WORK / f"{run_dir.name}-md{md}-{split}"
    if work.exists():
        shutil.rmtree(work)
    (work / "checkpoints").mkdir(parents=True)
    for name in ("best.pt", "best.pt.meta.json"):
        src = run_dir / "checkpoints" / name
        if not src.is_file():
            raise FileNotFoundError(f"缺少 {src}")
        (work / "checkpoints" / name).hardlink_to(src)   # 硬链接隔离，不触碰原 run
    cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    cfg.setdefault("evaluation", {})["smoothing_min_duration"] = md
    cfg["evaluation"]["visualize"] = False               # 只为省时间，不影响指标
    cfg.setdefault("data", {})["split_eval"] = split     # 该入口没有 --split，只能改配置
    cfg_path = work / "config.resolved.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return cfg_path, work / "checkpoints" / "best.pt"


def evaluate(cfg_path: Path, ckpt: Path, out_dir: Path) -> dict:
    """跑一次 benchmark.cli.eval，返回 metrics.summary 里的标量。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    env = {"PYTHONPATH": str(REPO / "framework"), "PATH": "/usr/bin:/bin"}
    log = out_dir / "eval.log"
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, "-m", "benchmark.cli.eval", "--config", str(cfg_path),
             "--ckpt", str(ckpt), "--out-dir", str(out_dir)],
            cwd=str(REPO), env={**env, "HOME": str(Path.home())}, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"    [WARN] eval 退出码 {proc.returncode}，见 {log}", file=sys.stderr)
    files = sorted(out_dir.glob("*.evaluation.json"))
    if not files:
        return {}
    summary = json.loads(files[0].read_text(encoding="utf-8"))["metrics"]["summary"]
    return {k: v["value"] for k, v in summary.items() if v.get("state") == "computed"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, help="在线臂 run 目录 glob（每个 run 一个 seed）")
    ap.add_argument("--md", nargs="+", type=int, required=True, help="要比较的门槛，如 25 10")
    ap.add_argument("--split", default="val", help="val（选参，默认）或 test（报数）")
    ap.add_argument("--keys", default="acc,edit", help="参与比较的指标")
    ap.add_argument("--json", default=None, help="把逐 seed 结果写成 JSON")
    args = ap.parse_args(argv)

    arms = [load_arm(p) for p in sorted(REPO.glob(args.arm))]
    if not arms:
        print(f"[ERROR] 未匹配到 run: {args.arm}", file=sys.stderr)
        return 1
    md0 = args.md[0]
    print(f"[online-md] {len(arms)} 个 seed × {len(args.md)} 个门槛，split={args.split}"
          f"；基准 = md={md0}")

    per_md: dict[int, dict[int, dict]] = {md: {} for md in args.md}
    for arm in arms:
        for md in args.md:
            cfg_path, ckpt = prepare(arm["dir"], md, args.split)
            out = WORK / "out" / f"{arm['dir'].name}-md{md}-{args.split}"
            metrics = evaluate(cfg_path, ckpt, out)
            per_md[md][arm["seed"]] = metrics
            print(f"  seed={arm['seed']:<4} md={md:<3} " +
                  "  ".join(f"{k}={metrics.get(k, float('nan')):.2f}" for k in args.keys.split(",")))

    keys = args.keys.split(",")
    print(f"\n=== 逐 seed 配对（vs md={md0}），n={len(arms)} ===")
    print(f"{'md':>5}{'n':>4}" + "".join(f"{k+' Δ':>10}{'p':>9}" for k in keys))
    for md in args.md[1:]:
        seeds = sorted(set(per_md[md]) & set(per_md[md0]))
        row = f"{md:>5}{len(seeds):>4}"
        for k in keys:
            a = [per_md[md0][s].get(k) for s in seeds]
            b = [per_md[md][s].get(k) for s in seeds]
            pair = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            if len(pair) < 2:
                row += f"{'n/a':>10}{'':>9}"
                continue
            x, y = np.array([p[0] for p in pair]), np.array([p[1] for p in pair])
            d = y - x
            p = wilcoxon(y, x).pvalue if np.any(d) else float("nan")
            row += f"{statistics.median(d):>+10.2f}{p:>9.4f}"
        print(row)
    print(f"\n  提示：n<16 时按本仓库纪律不可判 <1.5pp 的差异；"
          f"门槛应在 val 上选、再到 test 上报。")

    if args.json:
        Path(args.json).write_text(
            json.dumps({str(md): {str(s): v for s, v in d.items()} for md, d in per_md.items()},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  JSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
