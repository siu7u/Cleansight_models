"""单帧检测流水线（DetectionPipeline）。

一条完整的训练+评估单元：消费 cleansight-yolo-pipeline 产出的标准 YOLO 数据集
（images/labels/data.yaml），用 ultralytics 训练/验证，**只产事实结果**：mAP / P / R 逐类
三态指标 + 完整性，不含任何业务门槛、不判 PASS/FAIL、不设非零退出码。

检测是**单帧无状态**语义：流水线自持推理（ultralytics.val），单帧语义写成模块常量
``SINGLE_FRAME_SEMANTICS`` 直接写入结果；实时延迟标 N/A（离线检测不测实时延迟）。检测域
与时序域不共享任何数据/模型抽象，仅在统一结果与矩阵处汇合。
"""

from __future__ import annotations

import json

from ..core.checkpoint import load_meta, meta_path_for
from ..core.environment import now_stamp, set_seed
from ..core.envelope import EvaluationResult, MetricValue
from ..core.execution import PredictionOutput, format_params
from ..core.integrity import assert_checkpoint_config, check_result_complete
from ..core.run import RunContext
from .artifacts import build_prediction_artifact
from .metrics import build_detection_metrics
from .yolo import get_adapter

# 单帧无状态推理语义（写入 EvaluationResult.inference）。
SINGLE_FRAME_SEMANTICS = {
    "mode": "single_frame",
    "sees": "one_image",
    "stateless": True,
    "windowing": "none",
    "cold_start": "n/a",
    "reset": "per_image",
    "note": "单帧无状态检测，逐图独立推理",
}

_SPEC_LATENCY = "latency/single_tick_ms/v1"


def _na_performance(reason: str) -> dict[str, MetricValue]:
    """检测离线评估不测实时延迟：三个延迟指标统一标 N/A（不是 0、不是缺失）。"""
    return {
        "latency_mean_ms": MetricValue.not_applicable(reason, spec=_SPEC_LATENCY),
        "latency_median_ms": MetricValue.not_applicable(reason, spec=_SPEC_LATENCY),
        "latency_p95_ms": MetricValue.not_applicable(reason, spec=_SPEC_LATENCY),
    }


class DetectionPipeline:
    pipeline_name = "detection"

    def validate_config(self, cfg: dict) -> None:
        if "type" not in cfg.get("model", {}):
            raise ValueError("检测流水线 model 段需包含 type（如 yolo）")
        data = cfg.get("data", {})
        if "data_yaml" not in data:
            raise ValueError("检测流水线 data 段需包含 data_yaml（指向 YOLO 数据集的 data.yaml）")

    def train(self, cfg: dict, runs_dir: str, seed: int, device) -> str:
        set_seed(seed)
        model_cfg = cfg["model"]
        adapter = get_adapter(model_cfg["type"])
        train_cfg = cfg.get("train", {})

        run = RunContext(runs_dir, label=model_cfg["type"])
        run.save_config(cfg)
        run.save_env(device, seed=seed)

        data_yaml = cfg["data"]["data_yaml"]
        name = cfg["data"].get("name", "detect")
        best_pt, num_params, names, nc = adapter.train(
            weights=model_cfg.get("weights", "yolo11n.pt"),
            data_yaml=data_yaml,
            train_cfg=train_cfg,
            imgsz=model_cfg.get("imgsz", 640),
            device=device,
            project=str(run.checkpoints_dir),
            name=name,
        )

        # sidecar 重建/溯源元信息：让 YOLO 权重也能进异构矩阵、被 load_meta 校验。
        meta = {
            "type": model_cfg["type"],
            "pipeline": self.pipeline_name,
            "nc": nc,
            "names": names,
            "num_params": num_params,
            "dataset": name,
            "data_yaml": str(data_yaml),
            "model": model_cfg,
            "train": train_cfg,
            "trained_at": now_stamp(),
        }
        meta_path_for(best_pt).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[train] run_dir={run.dir}")
        print(f"[train] checkpoint={best_pt}")
        return str(best_pt)

    def predict(self, cfg: dict, ckpt: str, device) -> PredictionOutput:
        """运行检测验证和逐图推理，返回不含 framework 指标判分的事实。"""

        model_cfg = cfg["model"]
        try:
            meta = load_meta(ckpt)
            assert_checkpoint_config(meta, {"type": model_cfg["type"]})
        except FileNotFoundError:
            if not model_cfg.get("allow_missing_meta", False):
                raise
            meta = {
                "type": model_cfg["type"],
                "pipeline": self.pipeline_name,
                "dataset": cfg["data"].get("name", cfg["data"]["data_yaml"]),
                "data_yaml": cfg["data"]["data_yaml"],
                "model": model_cfg,
                "num_params": None,
                "source": "missing_meta_fallback",
            }

        adapter = get_adapter(model_cfg["type"])
        split = cfg["data"].get("eval_split", "val")
        val = adapter.val(
            weights=ckpt,
            data_yaml=cfg["data"]["data_yaml"],
            split=split,
            imgsz=model_cfg.get("imgsz", 640),
            device=device,
        )

        raw_predictions: dict = {}
        labels = val.get("names", {})
        errors: list[str] = []
        if hasattr(adapter, "predict"):
            try:
                raw = adapter.predict(
                    weights=ckpt,
                    data_yaml=cfg["data"]["data_yaml"],
                    split=split,
                    imgsz=model_cfg.get("imgsz", 640),
                    device=device,
                )
                raw_predictions = raw.get("items", {})
                labels = raw.get("labels", labels)
            except Exception as exc:
                errors.append(f"逐图预测生成失败: {type(exc).__name__}: {exc}")

        return PredictionOutput(
            model_type=model_cfg["type"],
            model_id=f"{model_cfg['type']}-{format_params(meta.get('num_params'))}",
            pipeline=self.pipeline_name,
            checkpoint=str(ckpt),
            dataset=cfg["data"].get("name", cfg["data"]["data_yaml"]),
            predictions=raw_predictions,
            labels=labels,
            feature_schema={"modality": "image", "imgsz": model_cfg.get("imgsz", 640)},
            inference_semantics=dict(SINGLE_FRAME_SEMANTICS),
            num_params=meta.get("num_params"),
            native_metrics=val,
            metadata={
                "split": split,
                "prediction_format": "class_confidence_xywhn",
                "data_yaml": str(cfg["data"]["data_yaml"]),
                "input_shape": ["B", 3, model_cfg.get("imgsz", 640), model_cfg.get("imgsz", 640)],
            },
            errors=errors,
        )

    def evaluate(self, cfg: dict, ckpt: str, device) -> EvaluationResult:
        """兼容评估入口：消费 ``predict`` 的事实输出并组装正式 EvaluationResult。"""

        output = self.predict(cfg, ckpt, device)
        metrics = build_detection_metrics(output.native_metrics)
        performance = _na_performance(reason="单帧检测评估不测实时延迟")

        result = EvaluationResult(
            model_type=output.model_type,
            model_id=output.model_id,
            pipeline=self.pipeline_name,
            checkpoint=output.checkpoint,
            dataset=output.dataset,
            feature_schema=output.feature_schema,
            metrics=metrics,
            performance=performance,
            inference_semantics=output.inference_semantics,
            num_params=output.num_params,
            timestamp=now_stamp(),
        )
        if output.predictions and cfg.get("evaluation", {}).get("save_predictions", True):
            result.pending_artifacts["predictions"] = build_prediction_artifact(
                output.predictions,
                output.labels,
                split=output.metadata["split"],
                prediction_format=output.metadata["prediction_format"],
            )
        elif output.errors:
            result.artifacts["predictions"] = {
                "state": "missing",
                "reason": output.errors[0],
            }
        result.integrity = check_result_complete(result)
        return result
