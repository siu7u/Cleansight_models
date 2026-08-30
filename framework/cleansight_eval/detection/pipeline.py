"""单帧检测流水线（DetectionPipeline）。

一条完整的训练+推理单元：消费 cleansight-yolo-pipeline 产出的标准 YOLO 数据集
（images/labels/data.yaml），用 ultralytics 训练并输出逐图预测事实。正式指标、完整性、
artifact 和报告全部由 benchmark 负责。

检测是**单帧无状态**语义：流水线自持模型执行，单帧语义写成模块常量
``SINGLE_FRAME_SEMANTICS`` 并随 ``PredictionOutput`` 交给 benchmark。检测域与时序域不共享
任何数据/模型抽象，只通过公共执行事实边界汇合。
"""

from __future__ import annotations

from pathlib import Path

from ..core.checkpoint import load_meta, write_meta
from ..core.environment import now_stamp, set_seed
from ..core.execution import PredictionOutput, format_params
from ..core.integrity import assert_checkpoint_config
from ..core.pipeline import Pipeline
from ..core.run import RunContext
from .yolo import get_adapter

# 单帧无状态推理语义（由 benchmark evaluator 写入 EvaluationResult.inference）。
SINGLE_FRAME_SEMANTICS = {
    "mode": "single_frame",
    "stateless": True,
}

class DetectionPipeline(Pipeline):
    pipeline_name = "detection"

    def validate_config(self, cfg: dict) -> None:
        if "type" not in cfg.get("model", {}):
            raise ValueError("检测流水线 model 段需包含 type（如 yolo）")
        if cfg.get("augmentation"):
            raise ValueError("augmentation.target_mask 仅支持 ActionMixed 时序流水线")
        data = cfg.get("data", {})
        if "data_yaml" not in data:
            raise ValueError("检测流水线 data 段需包含 data_yaml（指向 YOLO 数据集的 data.yaml）")

    def train(self, cfg: dict, runs_dir: str, seed: int, device, run_id: str | None = None) -> str:
        set_seed(seed)
        model_cfg = cfg["model"]
        adapter = get_adapter(model_cfg["type"])
        train_cfg = cfg.get("train", {})

        # resume 语义：ultralytics resume=True 从 self.ckpt_path（= model.weights）续训，
        # 且从 ckpt 恢复原 project/name 续写原目录。framework 侧因此复用原 run 目录，
        # 避免每次 resume 新建 runs/yolo-<新时间戳>。
        run_id = None
        if train_cfg.get("resume"):
            resume_ckpt = Path(str(model_cfg.get("weights", "")))
            # last.pt 形如 <run>/checkpoints/<group>/weights/last.pt → 向上 4 级是 run 目录
            if resume_ckpt.is_file() and "checkpoints" in resume_ckpt.parts:
                run_id = resume_ckpt.parents[3].name
                print(f"[train] resume: 复用原 run 目录 runs/{run_id}")

        run = RunContext(runs_dir, label=model_cfg["type"], run_id=run_id)
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
