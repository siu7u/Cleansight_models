#!/usr/bin/env python3
"""评测一个 3 分钟 CleanSight 端到端 benchmark case。

该脚本刻意与在线服务解耦，只评测导出的预测时间线。在线 workflow 只需要导出
如下格式的 JSON：

{
  "case_id": "clean_001",
  "result": "pass",
  "actions": [{"name": "Long_Brushing", "start_sec": 31, "end_sec": 88}],
  "alarms": [{"type": "missing_short_brushing", "message": "..."}]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "benchmark" / "e2e_3min" / "reports"


def load_yaml(path: Path) -> dict:
    """加载一个端到端 benchmark case YAML 文件。"""

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_prediction(path: Path | None) -> dict | None:
    """加载 workflow 导出的预测结果；未提供时返回 None 表示待接入。"""

    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def overlap_seconds(a: dict, b: dict) -> float:
    """返回两个动作时间段之间的重叠秒数。"""

    start = max(float(a["start_sec"]), float(b["start_sec"]))
    end = min(float(a["end_sec"]), float(b["end_sec"]))
    return max(0.0, end - start)


def score_case(case: dict, prediction: dict | None) -> dict:
    """用导出的端到端预测结果评测一个 3 分钟 case。

    prediction 格式是 Backend 或等价离线 workflow 导出的动作时间线 JSON。
    未提供 prediction 时输出 PENDING 报告而不是失败，方便先审阅 case 定义。
    """

    expected = case.get("expected", {})
    required = expected.get("required_actions", [])
    phases = expected.get("phases", [])
    allowed = float(expected.get("allowed_time_error_sec", 5))

    if prediction is None:
        return {
            "case_id": case["case_id"],
            "status": "PENDING",
            "reason": "缺少 prediction JSON；需要先跑线上或离线 workflow 导出动作时间线。",
            "action_recall": {},
            "phase_errors": [],
            "result_match": None,
            "alarms": [],
        }

    actions = prediction.get("actions", [])
    by_name = {}
    for action in actions:
        by_name.setdefault(action.get("name"), []).append(action)

    action_recall = {}
    for name in required:
        action_recall[name] = bool(by_name.get(name))

    phase_errors = []
    for phase in phases:
        candidates = by_name.get(phase["name"], [])
        if not candidates:
            phase_errors.append({"name": phase["name"], "matched": False, "reason": "未检出"})
            continue
        best = max(candidates, key=lambda item: overlap_seconds(phase, item))
        start_error = float(best["start_sec"]) - float(phase["start_sec"])
        end_error = float(best["end_sec"]) - float(phase["end_sec"])
        phase_errors.append(
            {
                "name": phase["name"],
                "matched": abs(start_error) <= allowed and abs(end_error) <= allowed,
                "start_error_sec": round(start_error, 3),
                "end_error_sec": round(end_error, 3),
            }
        )

    result_match = prediction.get("result") == expected.get("result")
    passed = result_match and all(action_recall.values()) and all(item.get("matched") for item in phase_errors)
    return {
        "case_id": case["case_id"],
        "status": "PASS" if passed else "FAIL",
        "action_recall": action_recall,
        "phase_errors": phase_errors,
        "result_match": result_match,
        "alarms": prediction.get("alarms", []),
    }


def write_report(case: dict, score: dict, out: Path) -> None:
    """为一个端到端 case 的评分结果写出易读的 Markdown 报告。"""

    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# 3 分钟端到端 Benchmark：{case['case_id']}",
        "",
        f"- 视频：`{case.get('video')}`",
        f"- 时长：{case.get('duration_sec')} 秒",
        f"- 结论：**{score['status']}**",
        "",
        "## 流程结论",
        "",
        f"- 期望：`{case.get('expected', {}).get('result')}`",
        f"- 是否一致：`{score.get('result_match')}`",
        "",
        "## 关键动作召回",
        "",
        "| 动作 | 是否检出 |",
        "| --- | --- |",
    ]
    recalls = score.get("action_recall") or {}
    if recalls:
        for name, ok in recalls.items():
            lines.append(f"| {name} | {'是' if ok else '否'} |")
    else:
        lines.append("| 待接入 | 待接入 |")

    lines += ["", "## 阶段时间误差", "", "| 阶段 | 是否匹配 | 起点误差秒 | 终点误差秒 |", "| --- | --- | ---: | ---: |"]
    for item in score.get("phase_errors") or []:
        lines.append(
            f"| {item['name']} | {'是' if item.get('matched') else '否'} | "
            f"{item.get('start_error_sec', 'NA')} | {item.get('end_error_sec', 'NA')} |"
        )
    if not score.get("phase_errors"):
        lines.append("| 待接入 | 待接入 | NA | NA |")

    if score.get("reason"):
        lines += ["", "## 待完成", "", f"- {score['reason']}"]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """解析命令行输入，评测一个 benchmark case，并写出报告。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="case yaml 路径")
    parser.add_argument("--prediction", help="workflow 导出的预测 JSON")
    parser.add_argument("--output", help="报告输出路径")
    args = parser.parse_args()

    case_path = Path(args.case)
    pred_path = Path(args.prediction) if args.prediction else None
    case = load_yaml(case_path)
    prediction = load_prediction(pred_path)
    score = score_case(case, prediction)

    out = Path(args.output) if args.output else OUT_DIR / f"{case['case_id']}.md"
    write_report(case, score, out)
    print(f"已写入 {out}")
    return 0 if score["status"] in {"PASS", "PENDING"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
