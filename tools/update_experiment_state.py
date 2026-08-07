#!/usr/bin/env python3
"""同步 EXPERIMENTS/STATE.json 与 LOG.md：扫描 runs/ 下的训练日志与结果 JSON。

新会话无缝续接用法：
    python tools/update_experiment_state.py
→ 读取 EXPERIMENTS/STATE.json 即可知道：已完成实验、进行中任务（日志路径）、下一步命令。

扫描范围：
  - runs/aug_*.log                每个增强实验的完整训练日志
  - runs/aug_compare_*.json       aug_experiments.py 的汇总结果
  - runs/*/weights/best.pt        训练权重
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
EXP = REPO / "EXPERIMENTS"
BEST_DIR = EXP / "best_weights"


def parse_val(log: Path) -> list[dict]:
    """从训练日志提取 [val] 结果行。"""
    out = []
    if not log.is_file():
        return out
    text = log.read_text(encoding="utf-8", errors="ignore")
    for m in re.finditer(r"\[val\] mAP50=([\d.]+) mAP50-95=([\d.]+) P=([\d.]+) R=([\d.]+)",
                         text):
        out.append({"map50": float(m.group(1)), "map50_95": float(m.group(2)),
                    "precision": float(m.group(3)), "recall": float(m.group(4))})
    return out


def scan() -> dict:
    state = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "repo": str(REPO),
        "experiments": [],
        "running": [],
        "notes": [],
    }
    # 按日志文件遍历
    now = datetime.now().timestamp()
    for log in sorted(RUNS.glob("aug_*.log")):
        vals = parse_val(log)
        name = log.name
        m = re.match(r"aug_g(\d)_(yolo11[ns])", name)
        group = f"group{int(m.group(1))}" if m and m.group(1) in {"1", "2"} else "?"
        model = m.group(2) if m else "?"
        # 运行中 ≈ 10 分钟内有写入（训练进度条持续刷新日志）
        try:
            mtime = log.stat().st_mtime
            running = (now - mtime) < 600
        except OSError:
            running = False
        entry = {
            "log": str(log.relative_to(REPO)),
            "group": group,
            "model": model,
            "presets_done": len(vals),
            "vals": vals,
            "running": running,
        }
        state["experiments"].append(entry)
    # 结果 JSON
    for jf in sorted(RUNS.glob("aug_compare_*.json")):
        state["experiments"].append({
            "result_json": str(jf.relative_to(REPO)),
            "note": "完整结果见 report_aug_results.py",
        })
    return state


def ensure_best_weights() -> None:
    """把各完成实验的 best.pt 复制为 <model>-<group>-<preset>-best.pt。

    只处理跑完 8 个 epoch 的 run（results.csv 数据行 >= 8），
    分组从 run 目录的 args.yaml 的 data 路径推断。
    """
    import shutil

    import yaml

    BEST_DIR.mkdir(parents=True, exist_ok=True)
    for best in sorted(RUNS.glob("yolo11?-*-480-8e-*/weights/best.pt")):
        run_dir = best.parent.parent
        parts = run_dir.name.split("-")  # yolo11s-default-480-8e-134407
        if len(parts) < 4:
            continue
        model, preset = parts[0], parts[1]
        results_csv = run_dir / "results.csv"
        if not results_csv.is_file():
            continue
        n_rows = len(results_csv.read_text(encoding="utf-8", errors="ignore").splitlines())
        if n_rows < 9:  # header + 8 epoch 才算完成
            print(f"[best] 跳过未完成: {run_dir.name}")
            continue
        try:
            args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
            data = str(args.get("data", ""))
            group = "g2" if "group2_small" in data else "g1"
        except Exception:
            group = "g?"
        dst = BEST_DIR / f"{model}-{group}-{preset}-best.pt"
        shutil.copy2(best, dst)
        print(f"[best] {dst.name}  <- {run_dir.name}")


def write(state: dict) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "STATE.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[state] 已写入 {EXP / 'STATE.json'}")
    # 追加 LOG.md
    log_path = EXP / "LOG.md"
    lines = [f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 状态同步",
             f"- 实验日志: {len(state['experiments'])} 条",
             f"- 见 STATE.json 详情"]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    state = scan()
    ensure_best_weights()
    write(state)
    # 打印摘要
    print(f"[scan] 找到 {len(state['experiments'])} 条实验记录")
    for e in state["experiments"]:
        if "vals" in e:
            status = "运行中" if e.get("running") else "已结束"
            print(f"  {e['log']}: {len(e['vals'])} 个预设完成 [{status}]")
        else:
            print(f"  {e.get('result_json')}")


if __name__ == "__main__":
    main()
