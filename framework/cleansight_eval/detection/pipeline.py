"""单帧检测流水线（DetectionPipeline）。

一条完整的训练+评估单元：消费 cleansight-yolo-pipeline 产出的标准 YOLO 数据集
（images/labels/data.yaml），用 ultralytics 训练/验证，**只产事实结果**：mAP / P / R 逐类
三态指标 + 完整性，不含任何业务门槛、不判 PASS/FAIL、不设非零退出码。

检测是**单帧无状态**语义：流水线自持推理（ultralytics.val），单帧语义写成模块常量
``SINGLE_FRAME_SEMANTICS`` 直接写入结果；实时延迟标 N/A（离线检测不测实时延迟）。检测域
与时序域不共享任何数据/模型抽象，仅在统一结果与矩阵处汇合。
"""

from __future__ import annotations

from ..core.checkpoint import load_meta, write_meta
from ..core.environment import now_stamp, set_seed
from ..core.execution import PredictionOutput, format_params
from ..core.integrity import assert_checkpoint_config
from ..core.run import RunContext
from .yolo import get_adapter

# 单帧无状态推理语义（由 benchmark evaluator 写入 EvaluationResult.inference）。
SINGLE_FRAME_SEMANTICS = {
    "mode": "single_frame",
    "stateless": True,
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
        run.write_status("running", stage="initializing")

        try:
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
            write_meta(best_pt, meta)
            last_pt = best_pt.with_name("last.pt")
            if last_pt.is_file():
                write_meta(last_pt, meta)
            results_csv = best_pt.parent.parent / "results.csv"
            training_curves = best_pt.parent.parent / "results.png"
            run.write_status(
                "succeeded",
                best_checkpoint=str(best_pt),
                last_checkpoint=str(last_pt) if last_pt.is_file() else None,
                history=str(results_csv) if results_csv.is_file() else None,
                training_curves=str(training_curves) if training_curves.is_file() else None,
            )
            print(f"[train] run_dir={run.dir}")
            print(f"[train] checkpoint={best_pt}")
            if training_curves.is_file():
                print(f"[train] training_curves={training_curves}")
            return str(best_pt)
        except Exception as exc:
            run.write_exception_status(exc)
            raise

    def predict(self, cfg: dict, ckpt: str, device) -> PredictionOutput:
        """运行检测验证和逐图推理，返回不含 framework 指标判分的事实。"""

        model_cfg = cfg["model"]
        evaluation_cfg = cfg.get("evaluation", {})
        mode = evaluation_cfg.get("mode", "formal")
        try:
            meta = load_meta(ckpt, require_schema=mode == "formal")
            assert_checkpoint_config(meta, {"type": model_cfg["type"]})
        except FileNotFoundError:
            if mode != "exploratory" or not model_cfg.get("allow_missing_meta", False):
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
        effective_parameters = {
            "conf": float(evaluation_cfg.get("conf", 0.001)),
            "iou": float(evaluation_cfg.get("iou", 0.7)),
            "imgsz": int(model_cfg.get("imgsz", 640)),
            "split": split,
            "max_det": int(evaluation_cfg.get("max_det", 300)),
            "agnostic_nms": bool(evaluation_cfg.get("agnostic_nms", False)),
        }
        val = adapter.val(
            weights=ckpt,
            data_yaml=cfg["data"]["data_yaml"],
            split=split,
            imgsz=model_cfg.get("imgsz", 640),
            device=device,
            conf=effective_parameters["conf"],
            iou=effective_parameters["iou"],
            max_det=effective_parameters["max_det"],
            agnostic_nms=effective_parameters["agnostic_nms"],
        )
        num_params = meta.get("num_params")
        if num_params is None:
            num_params = val.get("num_params")
        model_id = meta.get("model_id") or f"{model_cfg['type']}-{format_params(num_params)}"

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
                    conf=effective_parameters["conf"],
                    iou=effective_parameters["iou"],
                    max_det=effective_parameters["max_det"],
                    agnostic_nms=effective_parameters["agnostic_nms"],
                )
                raw_predictions = raw.get("items", {})
                labels = raw.get("labels", labels)
            except Exception as exc:
                errors.append(f"逐图预测生成失败: {type(exc).__name__}: {exc}")

        return PredictionOutput(
            model_type=model_cfg["type"],
            model_id=model_id,
            pipeline=self.pipeline_name,
            checkpoint=str(ckpt),
            dataset=cfg["data"].get("name", cfg["data"]["data_yaml"]),
            predictions=raw_predictions,
            labels=labels,
            feature_schema={"modality": "image", "imgsz": model_cfg.get("imgsz", 640)},
            inference_semantics=dict(SINGLE_FRAME_SEMANTICS),
            num_params=num_params,
            native_metrics=val,
            metadata={
                "split": split,
                "prediction_format": "class_confidence_xywhn",
                "data_yaml": str(cfg["data"]["data_yaml"]),
                "input_shape": ["B", 3, model_cfg.get("imgsz", 640), model_cfg.get("imgsz", 640)],
                "effective_parameters": effective_parameters,
                "metadata_source": meta.get("source", "bound_sidecar"),
            },
            errors=errors,
        )
