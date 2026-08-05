"""模型执行阶段的公共输出，不包含任何指标或报告语义。

``PredictionOutput`` 是 framework 运行模型后的边界对象：它只描述模型、数据、预测、
真值（若当前数据源提供）、原生验证结果和原始耗时样本。benchmark 消费这些事实计算
指标、生成 artifact 和报告；framework 不拥有评测入口或 EvaluationResult。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch


def format_params(n: int | None) -> str:
    """把参数量格式化成紧凑标注，缺失时不伪造数值。"""

    if not n:
        return "?"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


@dataclass
class PredictionOutput:
    """一次模型执行的事实输出。

    该对象故意不引用 ``MetricValue``、指标 spec、PASS/FAIL 或报告 schema。``predictions``
    和 ``targets`` 以 item id 为键，保留视频/图像边界；具体值由 pipeline 的 inference
    semantics 解释。
    """

    model_type: str
    model_id: str
    pipeline: str
    checkpoint: str
    dataset: str
    predictions: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    labels: Any = field(default_factory=list)
    feature_schema: dict = field(default_factory=dict)
    inference_semantics: dict = field(default_factory=dict)
    num_params: int | None = None
    native_metrics: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为仅含普通 Python 值的可序列化字典。"""

        return asdict(self)


def sample_callable_latency(
    callback: Callable[[], Any],
    device,
    *,
    warmup: int = 20,
    runs: int = 200,
    scope: str = "model_forward",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """执行 callback 并返回未汇总的逐次耗时样本（毫秒）。

    framework 只负责采样；mean/median/p95 等统计口径由后续评估层决定。CUDA 每次采样前
    等待 kernel 完成，避免只测到异步提交时间。
    """

    if runs <= 0:
        raise ValueError("runs 必须大于 0")
    if warmup < 0:
        raise ValueError("warmup 不能小于 0")

    device_type = getattr(device, "type", None) or str(device)

    def synchronize() -> None:
        if device_type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup):
            callback()
        synchronize()

        samples: list[float] = []
        for _ in range(runs):
            started = time.perf_counter()
            callback()
            synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)

    return {
        "scope": scope,
        "unit": "ms",
        "device": str(device),
        "warmup": warmup,
        "runs": runs,
        "samples_ms": samples,
        "context": dict(context or {}),
    }
