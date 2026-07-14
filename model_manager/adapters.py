"""把统一评估请求转换为各模型族的隔离子进程命令。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from model_manager.catalog import ROOT, ModelSpec


@dataclass(frozen=True)
class EvaluationRequest:
    """固定评估框架传给模型族 adapter 的请求。"""

    testset_id: str
    inference_mode: str
    device: str = "auto"
    max_videos: int | None = None
    max_frames: int | None = None
    append_card: bool = True


@dataclass(frozen=True)
class EvaluationCommand:
    """一个模型评估子进程的命令与工作目录。"""

    argv: list[str]
    cwd: Path


class EvaluationAdapter(Protocol):
    """新增模型族时需要实现的最小评估 adapter。"""

    def build(self, spec: ModelSpec, request: EvaluationRequest) -> EvaluationCommand:
        """构造模型族评估命令。"""


class TemporalEvaluationAdapter:
    """把时序模型接入统一逐帧与片段指标评估器。"""

    def build(self, spec: ModelSpec, request: EvaluationRequest) -> EvaluationCommand:
        if request.inference_mode != "raw_last_frame":
            raise ValueError(
                "单模型 evaluate 当前只接受 raw_last_frame；"
                "full_sequence/streaming 对比请使用 temporal_feed_mode benchmark"
            )
        argv = [
            sys.executable,
            str(ROOT / "tools" / "eval_temporal_detailed.py"),
            "--model-id",
            spec.id,
            "--testset",
            request.testset_id,
            "--device",
            request.device,
        ]
        if request.max_videos is not None:
            argv.extend(["--max-videos", str(request.max_videos)])
        if request.max_frames is not None:
            argv.extend(["--max-frames", str(request.max_frames)])
        if request.append_card and spec.card is not None:
            argv.extend(["--card", str(spec.card)])
        return EvaluationCommand(argv=argv, cwd=ROOT)


class YoloEvaluationAdapter:
    """把 YOLO checkpoint 接入统一 holdout benchmark。"""

    def build(self, spec: ModelSpec, request: EvaluationRequest) -> EvaluationCommand:
        if request.inference_mode != "detection":
            raise ValueError(f"YOLO 只接受 inference_mode=detection，收到 {request.inference_mode}")
        argv = [
            sys.executable,
            str(ROOT / "benchmark" / "single_model" / "run_yolo_benchmark.py"),
            "--model",
            spec.id,
            "--testset",
            request.testset_id,
        ]
        if spec.checkpoint is not None:
            argv.extend(["--weights", str(spec.checkpoint)])
        return EvaluationCommand(argv=argv, cwd=ROOT)


ADAPTERS: dict[str, EvaluationAdapter] = {
    "temporal_main": TemporalEvaluationAdapter(),
    "yolo_pipeline": YoloEvaluationAdapter(),
}


def evaluation_command(spec: ModelSpec, request: EvaluationRequest) -> EvaluationCommand:
    """经 adapter registry 构造命令，模型清单不再只是展示字段。"""

    try:
        adapter = ADAPTERS[spec.adapter]
    except KeyError as exc:
        raise ValueError(f"{spec.id} 使用未登记 adapter: {spec.adapter}") from exc
    return adapter.build(spec, request)
