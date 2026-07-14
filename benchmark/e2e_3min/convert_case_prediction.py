#!/usr/bin/env python3
"""把端到端 benchmark 的原始输入转换为标准 case 与 prediction 文件。

该模块只负责格式标准化，不执行模型推理、不计算指标。生成的文件可以直接交给
`run_e2e_benchmark.py` 评分。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_DIR = ROOT / "yolo-detection" / "pipeline" / "raw" / "exports"
DEFAULT_CASE_DIR = ROOT / "benchmark" / "e2e_3min" / "cases"
DEFAULT_OUTPUT_DIR = ROOT / "benchmark" / "e2e_3min" / "outputs"
DEFAULT_REQUIRED_ACTIONS = ["Long_Brushing", "Short_Brushing"]


@dataclass(frozen=True)
class TimeSegment:
    """表示一个动作时间段，单位统一为秒。"""

    name: str
    start_sec: float
    end_sec: float


def load_json(path: Path) -> Any:
    """读取 JSON 文件，返回原始 Python 对象。"""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    """以稳定缩进写出 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: dict) -> None:
    """以 benchmark case 需要的 YAML 格式写出文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def iter_tasks(payload: Any) -> list[dict]:
    """兼容 Label Studio 导出的 list 或带 tasks 字段的 dict。"""

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("tasks"), list):
            return [item for item in payload["tasks"] if isinstance(item, dict)]
        if isinstance(payload.get("data"), dict):
            return [payload]
    raise ValueError("无法识别 Label Studio 导出结构：期望 list 或包含 tasks 的 dict")


def iter_results(task: dict) -> list[dict]:
    """遍历一个 Label Studio task 中所有 annotation 的 result。"""

    results: list[dict] = []
    for annotation in task.get("annotations", []) or []:
        results.extend(item for item in annotation.get("result", []) or [] if isinstance(item, dict))
    return results


def task_video_name(task: dict) -> str:
    """从 Label Studio task 中提取视频文件名。"""

    video = task.get("data", {}).get("video") or task.get("data", {}).get("video_url") or ""
    return Path(str(video)).name


def infer_clip_meta(task: dict) -> tuple[float | None, float | None]:
    """尽量从 LS result 中推断标注帧数和视频秒数。"""

    for result in iter_results(task):
        value = result.get("value") or {}
        frames_count = value.get("framesCount")
        duration = value.get("duration")
        if frames_count and duration:
            return float(frames_count), float(duration)
    return None, None


def label_allowed(label: str, include_labels: set[str] | None, include_idle: bool) -> bool:
    """判断某个时间轴标签是否应该进入 benchmark case。"""

    if not include_idle and label == "Idle":
        return False
    return include_labels is None or label in include_labels


def value_to_seconds(raw: float, frames_count: float | None, duration: float | None, range_unit: str) -> float:
    """把 LS range 的 start/end 转成秒，支持 frame、second 和 auto 三种模式。"""

    if range_unit == "second":
        return raw
    if range_unit == "frame":
        if not frames_count or not duration:
            raise ValueError("range-unit=frame 需要 LS 导出中包含 framesCount 和 duration")
        return raw / (frames_count / duration)
    if range_unit != "auto":
        raise ValueError(f"未知 range-unit: {range_unit}")
    if frames_count and duration:
        return raw / (frames_count / duration)
    return raw


def collect_timeline_segments(
    task: dict,
    include_labels: set[str] | None,
    include_idle: bool,
    range_unit: str,
) -> list[TimeSegment]:
    """从 LS timelinelabels 中抽取动作段，并统一到秒级时间轴。"""

    frames_count, duration = infer_clip_meta(task)
    segments: list[TimeSegment] = []
    for result in iter_results(task):
        if result.get("type") != "timelinelabels":
            continue
        value = result.get("value") or {}
        labels = value.get("timelinelabels") or []
        if not labels:
            continue
        label = str(labels[0])
        if not label_allowed(label, include_labels, include_idle):
            continue
        for item in value.get("ranges", []) or []:
            start = value_to_seconds(float(item["start"]), frames_count, duration, range_unit)
            end = value_to_seconds(float(item["end"]), frames_count, duration, range_unit)
            if end > start:
                segments.append(TimeSegment(label, round(start, 3), round(end, 3)))
    return sorted(segments, key=lambda item: (item.start_sec, item.end_sec, item.name))


def build_case(
    task: dict,
    segments: list[TimeSegment],
    case_id: str,
    result: str,
    required_actions: list[str],
    allowed_time_error_sec: float,
) -> dict:
    """把一个 LS task 和动作段组装成 e2e benchmark case。"""

    _, duration = infer_clip_meta(task)
    inferred_duration = max((item.end_sec for item in segments), default=0.0)
    phases = [
        {"name": item.name, "start_sec": item.start_sec, "end_sec": item.end_sec}
        for item in segments
    ]
    return {
        "case_id": case_id,
        "source_task_id": task.get("id"),
        "video": task_video_name(task),
        "duration_sec": round(float(duration if duration is not None else inferred_duration), 3),
        "expected": {
            "result": result,
            "required_actions": required_actions,
            "phases": phases,
            "allowed_time_error_sec": allowed_time_error_sec,
        },
    }


def unique_case_id(base: str, used: set[str]) -> str:
    """保证同一批转换里的 case_id 不重复。"""

    candidate = base or "case"
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def case_id_from_task(task: dict, prefix: str | None, used: set[str]) -> str:
    """根据视频名或 task id 生成稳定 case_id。"""

    if prefix:
        base = f"{prefix}_{task.get('id') or Path(task_video_name(task)).stem or 'case'}"
    else:
        base = Path(task_video_name(task)).stem or f"task_{task.get('id', 'case')}"
    return unique_case_id(base.replace(" ", "_"), used)


def convert_cases(args: argparse.Namespace) -> int:
    """把一个或多个 LS 导出 JSON 转成 benchmark case YAML。"""

    exports = args.export or sorted(DEFAULT_EXPORT_DIR.glob("*.json"))
    if not exports:
        raise SystemExit(f"没有找到 Label Studio 导出 JSON: {DEFAULT_EXPORT_DIR}")

    include_labels = set(args.label) if args.label else None
    required_actions = args.required_action or DEFAULT_REQUIRED_ACTIONS
    used: set[str] = set()
    written: list[Path] = []

    for export_path in exports:
        for task in iter_tasks(load_json(export_path)):
            segments = collect_timeline_segments(task, include_labels, args.include_idle, args.range_unit)
            if not segments:
                continue
            case_id = case_id_from_task(task, args.case_prefix, used)
            case = build_case(
                task=task,
                segments=segments,
                case_id=case_id,
                result=args.result,
                required_actions=required_actions,
                allowed_time_error_sec=args.allowed_time_error_sec,
            )
            out = args.out_dir / f"{case_id}.yaml"
            write_yaml(out, case)
            written.append(out)

    if not written:
        raise SystemExit("没有从 LS 导出中找到可转换的 timelinelabels 时间段")
    for path in written:
        print(f"已写入 case: {path}")
    return 0


def first_present(mapping: dict, keys: tuple[str, ...]) -> Any:
    """按候选字段顺序取第一个存在且非空的值。"""

    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def normalize_action(item: dict, fps: float | None) -> dict | None:
    """把常见动作段字段归一化为 scorer 需要的 name/start_sec/end_sec。"""

    name = first_present(item, ("name", "label", "action", "phase", "class"))
    if not name:
        return None

    start = first_present(item, ("start_sec", "start_time", "start"))
    end = first_present(item, ("end_sec", "end_time", "end"))
    if start is None and item.get("start_frame") is not None and fps:
        start = float(item["start_frame"]) / fps
    if end is None and item.get("end_frame") is not None and fps:
        end = float(item["end_frame"]) / fps
    if start is None or end is None:
        return None

    start_sec = float(start)
    end_sec = float(end)
    if end_sec <= start_sec:
        return None
    return {"name": str(name), "start_sec": round(start_sec, 3), "end_sec": round(end_sec, 3)}


def extract_action_items(payload: Any) -> list[dict]:
    """从常见 prediction/segment/action 包装结构中提取动作列表。"""

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        raise ValueError("prediction 源文件必须是 dict 或 list")
    for key in ("actions", "segments", "phases", "events"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    raise ValueError("prediction 源文件缺少 actions/segments/phases/events 列表")


def convert_prediction(args: argparse.Namespace) -> int:
    """把动作时间线 JSON 归一化为 e2e scorer 的 prediction JSON。"""

    payload = load_json(args.actions_json)
    source = payload if isinstance(payload, dict) else {}
    actions = [item for item in (normalize_action(item, args.fps) for item in extract_action_items(payload)) if item]
    if not actions:
        raise SystemExit("没有从输入文件中找到可转换的动作时间段")

    case_id = args.case_id or source.get("case_id") or args.actions_json.stem.replace(".prediction", "")
    prediction = {
        "case_id": case_id,
        "result": args.result or source.get("result") or "pass",
        "actions": sorted(actions, key=lambda item: (item["start_sec"], item["end_sec"], item["name"])),
        "alarms": source.get("alarms", []),
    }

    out = args.out or DEFAULT_OUTPUT_DIR / f"{case_id}.prediction.json"
    write_json(out, prediction)
    print(f"已写入 prediction: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """创建 case/prediction 转换命令行解析器。"""

    parser = argparse.ArgumentParser(description="转换 e2e benchmark case 与 prediction 输入")
    subparsers = parser.add_subparsers(dest="command", required=True)

    case_parser = subparsers.add_parser("case-from-ls", help="从 Label Studio timelinelabels 生成 case YAML")
    case_parser.add_argument("--export", type=Path, action="append", help="LS 导出 JSON；可重复传入")
    case_parser.add_argument("--out-dir", type=Path, default=DEFAULT_CASE_DIR, help="case YAML 输出目录")
    case_parser.add_argument("--case-prefix", help="case_id 前缀；默认使用视频文件名")
    case_parser.add_argument("--label", action="append", help="只转换指定动作标签；可重复传入")
    case_parser.add_argument("--include-idle", action="store_true", help="保留 Idle 时间段")
    case_parser.add_argument("--required-action", action="append", help="期望必须召回的动作；默认使用刷洗动作")
    case_parser.add_argument("--result", default="pass", help="case 期望流程结论")
    case_parser.add_argument("--allowed-time-error-sec", type=float, default=5.0, help="阶段起止时间容忍误差")
    case_parser.add_argument(
        "--range-unit",
        choices=("auto", "frame", "second"),
        default="auto",
        help="LS ranges 单位；auto 有 framesCount/duration 时按帧号换算，否则按秒保留",
    )
    case_parser.set_defaults(func=convert_cases)

    pred_parser = subparsers.add_parser("prediction-from-actions", help="从动作段 JSON 生成 prediction JSON")
    pred_parser.add_argument("--actions-json", type=Path, required=True, help="后端或离线 workflow 导出的动作段 JSON")
    pred_parser.add_argument("--out", type=Path, help="prediction JSON 输出路径")
    pred_parser.add_argument("--case-id", help="覆盖 prediction 中的 case_id")
    pred_parser.add_argument("--result", help="覆盖 prediction 中的流程结论")
    pred_parser.add_argument("--fps", type=float, help="输入只有 start_frame/end_frame 时用于换算秒")
    pred_parser.set_defaults(func=convert_prediction)

    return parser


def main() -> int:
    """命令行入口。"""

    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
