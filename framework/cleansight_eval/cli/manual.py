"""手动训练 CLI：python -m framework.cleansight_eval.cli.manual。

在终端独立管理模型训练（后台启动、进度、恢复、日志、评测、体检），不依赖
agent 会话。训练/评测执行仍走 framework 的 ``cli.train`` 与 benchmark 的
``cli.eval``（进程级调用），本入口只做进程与 run 管理薄壳。

用法（仓库根执行）:
    # 前台训练（终端挂着看进度，Ctrl+C 中断）
    python -m framework.cleansight_eval.cli.manual start --model yolo11s --group group1_large

    # 后台训练（脱离终端，日志写 runs/manual/）
    python -m framework.cleansight_eval.cli.manual start --model yolo11s --group group1_large --bg

    # 指定配置训练（进阶）
    python -m framework.cleansight_eval.cli.manual start \
        --config framework/experiments/yolo-clean-large.yaml -S train.epochs=200

    # 查看进度
    python -m framework.cleansight_eval.cli.manual status
    python -m framework.cleansight_eval.cli.manual status --run runs/yolo-<ts> --json

    # 中断后从 last.pt 恢复
    python -m framework.cleansight_eval.cli.manual resume [--run <dir>] [--bg]

    # 评测最新 best.pt（走 benchmark.cli.eval）
    python -m framework.cleansight_eval.cli.manual eval [--run <dir>]

    # 训练日志
    python -m framework.cleansight_eval.cli.manual logs [-f] [-n 50]

    # 环境/数据/训练一键检查
    python -m framework.cleansight_eval.cli.manual doctor
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS = REPO_ROOT / "runs"
EXPERIMENTS = REPO_ROOT / "framework" / "experiments"
PYTHON = sys.executable


# ── 子命令 ──────────────────────────────────────────────


def _cmd_train(argv) -> int:
    """start: 前台或后台启动训练（复用 cli.train）。"""

    p = argparse.ArgumentParser(prog="manual start", description="启动训练")
    p.add_argument("--model", default=None, help="模型别名（yolo11s/gru/...；--list-models 查看）")
    p.add_argument("--group", default=None, help="YOLO 分组")
    p.add_argument("--config", default=None, help="实验配置 YAML（与 --model 二选一）")
    p.add_argument("--bg", action="store_true", help="后台运行（脱离终端，日志写 runs/manual/）")
    p.add_argument("--runs-dir", default=str(RUNS), help="运行输出根目录")
    p.add_argument("-S", "--set", action="append", default=[], metavar="KEY=VALUE",
                   help="配置覆盖，可多次")
    args = p.parse_args(argv)

    if not args.model and not args.config:
        print("[manual] 需要 --model <别名> 或 --config <yaml>（cli.train --list-models 查看别名）")
        return 2

    cmd = [PYTHON, "-m", "framework.cleansight_eval.cli.train", "--runs-dir", args.runs_dir]
    if args.model:
        cmd += ["--model", args.model]
    if args.group:
        cmd += ["--group", args.group]
    if args.config:
        cmd += ["--config", args.config]
    for s in args.set:
        cmd += ["-S", s]

    print(f"[manual] 训练命令: {' '.join(cmd)}")

    if not args.bg:
        return subprocess.call(cmd, cwd=REPO_ROOT)

    log_dir = RUNS / "manual"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train-{_stamp()}.log"
    with open(log_file, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    print(f"[manual] 后台训练已启动 PID={proc.pid}")
    print(f"  日志: {log_file}")
    print(f"  进度: python -m framework.cleansight_eval.cli.manual status")
    print(f"  跟踪: python -m framework.cleansight_eval.cli.manual logs -f")
    return 0


def _latest_run() -> Path | None:
    """找最新的 YOLO run 目录（runs/yolo-*）。"""

    candidates = sorted(RUNS.glob("yolo-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _run_checkpoints(run_dir: Path) -> Path:
    """YOLO run 的 checkpoints 目录（checkpoints/<group>/weights/）。"""

    for group_dir in (run_dir / "checkpoints").glob("*/"):
        weights = group_dir / "weights"
        if (weights / "best.pt").is_file():
            return weights
    return run_dir / "checkpoints"


def _read_resolved(run_dir: Path) -> dict:
    """读取 run 的 config.resolved.json（容错）。"""

    resolved = run_dir / "config.resolved.json"
    if not resolved.is_file():
        return {}
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _group_to_config(data_name: str) -> Path | None:
    """YOLO 组名 → 实验配置路径。"""

    if data_name == "group1_large":
        return EXPERIMENTS / "yolo-clean-large.yaml"
    if data_name == "group2_small":
        return EXPERIMENTS / "yolo-clean-small.yaml"
    return None


def _cmd_status(argv) -> int:
    """status: 显示训练进度（epoch、最新指标、进程是否存活）。"""

    p = argparse.ArgumentParser(prog="manual status", description="查看训练进度")
    p.add_argument("--run", default=None, help="run 目录；默认最新")
    p.add_argument("--json", action="store_true", help="输出 JSON")
    args = p.parse_args(argv)

    run_dir = Path(args.run) if args.run else _latest_run()
    if run_dir is None or not run_dir.is_dir():
        print("[manual] 未找到 run 目录（runs/yolo-*）。先 start。")
        return 1

    weights = _run_checkpoints(run_dir)
    results_csv = weights.parent / "results.csv"
    if not results_csv.is_file():
        results_csv = run_dir / "results.csv"
    status_path = run_dir / "status.json"

    info = {"run": str(run_dir), "checkpoints": str(weights)}
    if status_path.is_file():
        try:
            st = json.loads(status_path.read_text(encoding="utf-8"))
            info["state"] = st.get("state")
        except (json.JSONDecodeError, OSError):
            pass

    epochs_done = None
    latest = {}
    if results_csv.is_file():
        lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) >= 2:
            header = lines[0].split(",")
            epochs_done = len(lines) - 1
            values = lines[-1].split(",")
            latest = {header[i]: values[i] for i in range(min(len(header), len(values)))}

    info["epochs_done"] = epochs_done
    info["latest"] = {k: latest.get(k) for k in
                      ("epoch", "metrics/precision(B)", "metrics/recall(B)",
                       "metrics/mAP50(B)", "metrics/mAP50-95(B)")}

    last_pt = weights / "last.pt"
    if last_pt.is_file():
        age_min = abs((os.path.getmtime(last_pt) - _now()) / 60)
        info["last_pt_age_min"] = round(age_min, 1)
        info["training_alive"] = age_min < 15

    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    print(f"run:      {info['run']}")
    print(f"state:    {info.get('state', '?')}")
    print(f"epochs:   {epochs_done or 0} 完成")
    if info.get("training_alive") is not None:
        alive = "✅ 训练中" if info["training_alive"] else "❌ 疑似停止（last.pt 未更新）"
        print(f"alive:    {alive}（last.pt {info['last_pt_age_min']} 分钟前更新）")
    if latest:
        print(f"最新指标: P={latest.get('metrics/precision(B)','-')} "
              f"R={latest.get('metrics/recall(B)','-')} "
              f"mAP50={latest.get('metrics/mAP50(B)','-')} "
              f"mAP50-95={latest.get('metrics/mAP50-95(B)','-')}")
    print(f"weights:  {weights}")
    return 0


def _cmd_resume(argv) -> int:
    """resume: 从 last.pt 恢复训练。"""

    p = argparse.ArgumentParser(prog="manual resume", description="从中断处恢复训练")
    p.add_argument("--run", default=None, help="run 目录；默认最新")
    p.add_argument("--model", default=None, help="模型别名（默认从 run 配置推导）")
    p.add_argument("--bg", action="store_true", help="后台恢复")
    args = p.parse_args(argv)

    run_dir = Path(args.run) if args.run else _latest_run()
    if run_dir is None or not run_dir.is_dir():
        print("[manual] 未找到 run 目录。")
        return 1
    weights = _run_checkpoints(run_dir)
    last_pt = weights / "last.pt"
    if not last_pt.is_file():
        print(f"[manual] 找不到 {last_pt}，无法恢复。")
        return 1

    cfg = _read_resolved(run_dir)
    data_name = (cfg.get("data") or {}).get("name")
    weights_name = (cfg.get("model") or {}).get("weights", "")
    model = args.model
    if not model and weights_name:
        model = ("yolo11n" if "n.pt" in weights_name else
                 "yolo11s" if "s.pt" in weights_name else
                 "yolo11m" if "m.pt" in weights_name else "yolo")

    config_path = _group_to_config(data_name) if data_name else None

    cmd = [PYTHON, "-m", "framework.cleansight_eval.cli.train"]
    if config_path and config_path.is_file():
        cmd += ["--config", str(config_path)]
    elif model:
        cmd += ["--model", model]
    else:
        print("[manual] 无法推导原配置，请手动传 --model 或 --config。")
        return 1
    cmd += ["--resume", str(last_pt), "--force"]
    if data_name:
        cmd += ["--group", data_name]

    if not args.bg:
        print(f"[manual] 前台恢复: {' '.join(cmd)}")
        return subprocess.call(cmd, cwd=REPO_ROOT)

    log_dir = RUNS / "manual"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"resume-{_stamp()}.log"
    with open(log_file, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    print(f"[manual] 后台恢复已启动 PID={proc.pid}，日志: {log_file}")
    return 0


def _cmd_eval(argv) -> int:
    """eval: 评测最新 run 的 best.pt（走 benchmark.cli.eval）。"""

    p = argparse.ArgumentParser(prog="manual eval", description="评测最新训练的 best.pt")
    p.add_argument("--run", default=None, help="run 目录；默认最新")
    args = p.parse_args(argv)

    run_dir = Path(args.run) if args.run else _latest_run()
    if run_dir is None or not run_dir.is_dir():
        print("[manual] 未找到 run 目录。")
        return 1
    weights = _run_checkpoints(run_dir)
    best_pt = weights / "best.pt"
    if not best_pt.is_file():
        print(f"[manual] 找不到 {best_pt}。")
        return 1

    cfg = _read_resolved(run_dir)
    data_name = (cfg.get("data") or {}).get("name")
    config_path = _group_to_config(data_name) if data_name else None
    if config_path is None or not config_path.is_file():
        print("[manual] 无法推导评测配置，请手动指定：")
        print(f"  python -m benchmark.cli.eval --config <yaml> --ckpt {best_pt}")
        return 1

    cmd = [PYTHON, "-m", "benchmark.cli.eval", "--config", str(config_path),
           "--ckpt", str(best_pt)]
    print(f"[manual] 评测: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=REPO_ROOT)


def _cmd_logs(argv) -> int:
    """logs: 查看后台训练日志。"""

    p = argparse.ArgumentParser(prog="manual logs", description="查看训练日志")
    p.add_argument("-f", "--follow", action="store_true", help="跟踪模式")
    p.add_argument("-n", "--lines", type=int, default=50, help="显示行数")
    args = p.parse_args(argv)

    log_dir = RUNS / "manual"
    if not log_dir.is_dir():
        print("[manual] 无日志目录（未用 --bg 启动过）。")
        return 1
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        print("[manual] 无日志文件。")
        return 1
    log_file = logs[0]
    print(f"日志: {log_file}")
    if args.follow:
        return subprocess.call(["tail", "-f", str(log_file)])
    return subprocess.call(["tail", "-n", str(args.lines), str(log_file)])


def _cmd_doctor(argv) -> int:
    """doctor: 环境 + 数据 + 训练状态一键检查。"""

    argparse.ArgumentParser(prog="manual doctor", description="环境/数据/训练检查").parse_args(argv)

    print("== 环境 ==")
    env_code = subprocess.call([PYTHON, str(REPO_ROOT / "tools/team_env.py")], cwd=REPO_ROOT)
    print()
    print("== 数据 ==")
    data_code = subprocess.call(
        [PYTHON, "-m", "framework.cleansight_eval.cli.dataset", "--check"], cwd=REPO_ROOT)
    print()
    print("== 最新训练 ==")
    _cmd_status(["--json"])
    return 0 if env_code == 0 and data_code == 0 else 1


def _now() -> float:
    import time

    return time.time()


def _stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S")


HELP_TEXT = """\
CleanSight 手动训练 CLI（framework 层；训练执行走 cli.train，评测走 benchmark.cli.eval）

用法:
  python -m framework.cleansight_eval.cli.manual start --model yolo11s --group group1_large [--bg]
  python -m framework.cleansight_eval.cli.manual start --config <yaml> [-S key=value] [--bg]
  python -m framework.cleansight_eval.cli.manual status [--run <dir>] [--json]
  python -m framework.cleansight_eval.cli.manual resume [--run <dir>] [--bg]
  python -m framework.cleansight_eval.cli.manual eval [--run <dir>]
  python -m framework.cleansight_eval.cli.manual logs [-f] [-n 50]
  python -m framework.cleansight_eval.cli.manual doctor

子命令:
  start    启动训练（前台/后台；--bg 脱离终端，日志写 runs/manual/）
  status   查看最新/指定 run 的进度（epoch、最新 P/R/mAP、进程存活）
  resume   从中断处 last.pt 恢复训练（自动推导原配置）
  eval     评测最新 run 的 best.pt（走 benchmark.cli.eval）
  logs     查看后台训练日志（-f 跟踪）
  doctor   环境/数据/训练状态一键检查
"""


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(HELP_TEXT)
        return 0

    command, rest = argv[0], argv[1:]
    handlers = {
        "start": _cmd_train,
        "status": _cmd_status,
        "resume": _cmd_resume,
        "eval": _cmd_eval,
        "logs": _cmd_logs,
        "doctor": _cmd_doctor,
    }
    if command not in handlers:
        print(f"未知子命令 '{command}'。可用: {', '.join(handlers)}")
        return 2
    return handlers[command](rest)


if __name__ == "__main__":
    raise SystemExit(main())
