"""把**已训练的 run** 在另一个 split 上复评（默认 val），用于执行"test 涨、val 不支持不计交付"这条纪律。

背景：`benchmark.cli.eval` 只评配置里 `data.split_eval` 指定的那个 split（通常是 test），
而本仓库的判读纪律要求候选在 **val** 侧也成立（[`docs/EVAL.md`](../docs/EVAL.md) §9.2）。
本工具把"同一批 checkpoint 在 val 上复评一遍"做成一条命令：

1. 对每个 run 复制它的 `config.resolved.json`（= 源配置 + 全部 `-S` 覆盖）并只改 `data.split_eval`；
2. 把 `checkpoints/best.pt`（与 `.meta.json`）以**硬链接**放到隔离目录 —— 不能直接对原
   checkpoint 评测：`benchmark.cli.eval` 会把 `best.eval.md` / `EVALUATION_REPORT.md`
   写回 checkpoint 所在目录，**覆盖该 run 原本的 test 报告**；
3. 对新配置跑 `benchmark.cli.eval`，评测产物落在 `--out-root/<run 名>/`。

用法：

    # 把一个臂的 8 个 run 全部在 val 上复评
    python tools/eval_runs_on_split.py --run "runs/v41/feat-roi-v5/h128/mstcn2-*" \\
        --split val --out-root tmp/val_evals --work-root tmp/val_runs --workers 3

    # 复评结束后可直接用投票探针读产物（run 目录里已有 artifacts/ 与 env.json）
    python tools/probe_cross_contract_vote.py --source val \\
        --arm feat-roi-v5="tmp/val_runs/feat-roi-v5-*"

注意：`--work-root` 与 `--out-root` 都在 scratch（`tmp/`）下，可随时删除；本工具不修改原 run。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = str(Path(sys.executable))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="把已训练 run 在指定 split 上复评")
    parser.add_argument("--run", action="append", required=True, metavar="GLOB",
                        help="run 目录 glob（可多次）；目录内需有 checkpoints/best.pt 与 env.json")
    parser.add_argument("--split", default="val", help="复评用的 split（写入 data.split_eval）")
    parser.add_argument("--out-root", default="tmp/eval-split", help="评测产物根目录")
    parser.add_argument("--work-root", default="tmp/eval-split-runs",
                        help="隔离的 run 目录根（放硬链接 checkpoint 与 split 变体配置）")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="运行 benchmark.cli.eval 的解释器")
    return parser.parse_args(argv)


def prepare(run: Path, split: str, work_root: Path) -> tuple[Path, Path] | None:
    """建隔离 run 目录（硬链接 checkpoint + split 变体配置），返回 (配置路径, checkpoint 路径)。"""

    if not (run / "checkpoints" / "best.pt").is_file() or not (run / "config.resolved.json").is_file():
        return None
    name = f"{run.parent.name}-{run.name}" if run.parent.name != "h128" else f"{run.parent.parent.name}-{run.name}"
    base = work_root / name
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)
    for filename in ("config.resolved.json", "env.json"):
        source = run / filename
        if source.is_file() and not (base / filename).exists():
            (base / filename).write_bytes(source.read_bytes())
    for filename in ("best.pt", "best.pt.meta.json"):
        source = run / "checkpoints" / filename
        target = base / "checkpoints" / filename
        if source.is_file() and not target.exists():
            os.link(source, target)  # 硬链接：内容共享、报告写在隔离目录，原 run 不被改动
    config = json.loads((base / "config.resolved.json").read_text())
    config["data"]["split_eval"] = split
    config.setdefault("evaluation", {})["save_predictions"] = True
    variant = base / f"config.{split}.json"
    variant.write_text(json.dumps(config, ensure_ascii=False, indent=1))
    return variant, base / "checkpoints" / "best.pt"


def evaluate(run: Path, args) -> tuple[str, Path | None, int]:
    prepared = prepare(run, args.split, REPO / args.work_root)
    if prepared is None:
        return run.name, None, -1
    config, checkpoint = prepared
    name = config.parent.name
    out_dir = REPO / args.out_root / name
    existing = sorted(out_dir.glob("*.evaluation.json")) if out_dir.is_dir() else []
    if existing:
        return name, existing[-1], 0
    out_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [args.python, "-m", "benchmark.cli.eval", "--config", str(config),
         "--ckpt", str(checkpoint), "--out-dir", str(out_dir)],
        cwd=REPO, capture_output=True, text=True)
    found = sorted(out_dir.glob("*.evaluation.json"))
    if process.returncode != 0:
        sys.stderr.write(process.stdout[-2000:] + process.stderr[-2000:])
    return name, (found[-1] if found else None), process.returncode


def main(argv=None) -> int:
    args = parse_args(argv)
    runs: list[Path] = []
    for pattern in args.run:
        runs.extend(sorted(p for p in REPO.glob(pattern) if p.is_dir()))
    runs = [run for run in runs if (run / "checkpoints" / "best.pt").is_file()]
    if not runs:
        print("[eval-split] 未匹配到含 checkpoints/best.pt 的 run")
        return 2
    print(f"[eval-split] {len(runs)} 个 run → split={args.split}，{args.workers} 并发")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, path, code in pool.map(lambda run: evaluate(run, args), runs):
            summary = {}
            if path and path.is_file():
                summary = json.loads(path.read_text())["metrics"]["summary"]
            rows.append((name, code, summary.get("acc", {}).get("value"),
                         summary.get("edit", {}).get("value")))
            print(f"[eval-split] {name:<34} rc={code} acc={rows[-1][2]} edit={rows[-1][3]}", flush=True)

    usable = [row for row in rows if row[1] == 0 and row[2] is not None]
    if usable:
        import statistics

        print(f"\n[eval-split] split={args.split} 有效 {len(usable)}/{len(rows)}："
              f"acc 中位 {statistics.median(r[2] for r in usable):.2f}"
              f"（摆幅 {min(r[2] for r in usable):.2f}~{max(r[2] for r in usable):.2f}）、"
              f"edit 中位 {statistics.median(r[3] for r in usable):.2f}"
              f"（摆幅 {min(r[3] for r in usable):.2f}~{max(r[3] for r in usable):.2f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
