"""YOLO 检测适配器（封装 ultralytics，检测流水线专属）。

ultralytics 自持训练/验证，本适配器只暴露 ``train`` / ``val`` 两个方法，由
``get_adapter(model_type)`` 取用。检测与时序两域故意不强行统一为同一套契约。

本适配器同时充当**检测的 data loader**：检测的输入就是**图像**、语义是**单帧无状态**，
ultralytics 从 ``data.yaml`` 一次性读入 images/labels 并自持批处理——无需另写 loader。

ultralytics/torch 为重依赖，全部在方法内部 import，使仅做数据/纯逻辑的场景
（如检测指标单元测试、注入假 adapter 的冒烟）无需安装它们。
"""

from __future__ import annotations

import os
from pathlib import Path

from ..core.config import YOLO_TRAIN_HPARAMS


def _ul_device(device) -> str:
    """torch.device -> ultralytics 的 device 参数字符串。"""
    t = getattr(device, "type", None) or str(device)
    if t == "cuda":
        idx = getattr(device, "index", None)
        return str(idx) if idx is not None else "0"
    return t  # "mps" / "cpu"


class YoloAdapter:
    model_type = "yolo"

    def train(self, weights, data_yaml, train_cfg: dict, imgsz: int, device, project, name):
        """训练 YOLO，返回 (best_pt, num_params, names, nc)。

        ultralytics 自行把权重写到 ``project/name/weights/best.pt``；本方法不接管
        权重落盘，由 DetectionTask 另写 sidecar 元信息。

        ``train_cfg`` 中登记的 ``YOLO_TRAIN_HPARAMS`` 超参（cos_lr、增强、freeze 等）
        会透传给 ``model.train()``；白名单外的键不转发并告警，避免拼写错误被静默忽略。

        训练默认关闭 ultralytics 在线检查（``YOLO_OFFLINE=true``）：PyPI 更新检查在
        弱网下会在训练开始前长时间挂起；用户可用 ``YOLO_OFFLINE=false`` 显式恢复。
        """
        # ONLINE 是 ultralytics import 时计算的模块常量，因此必须在 import 之前设置；
        # setdefault 尊重用户显式设置的环境变量。
        os.environ.setdefault("YOLO_OFFLINE", "true")
        from ultralytics import YOLO

        model = YOLO(str(weights))
        # epochs/batch/patience 由下方显式传参，其余登记的 YOLO 超参走白名单透传。
        hparams = {k: v for k, v in train_cfg.items() if k in YOLO_TRAIN_HPARAMS}
        ignored = sorted(set(train_cfg) - YOLO_TRAIN_HPARAMS
                         - {"epochs", "batch", "patience", "resume"})
        if ignored:
            print(f"[yolo.train] 忽略未登记的训练超参: {ignored}")
        # project 必须传绝对路径：ultralytics 对相对 project 不照单全收，会把它拼到
        # 自身 settings 的 runs_dir（默认 runs/detect）下，导致产物落到预期之外的目录。
        model.train(
            data=str(data_yaml),
            epochs=train_cfg.get("epochs", 100),
            imgsz=imgsz,
            batch=train_cfg.get("batch", 16),
            patience=train_cfg.get("patience", 20),
            device=_ul_device(device),
            project=str(Path(project).resolve()),
            name=str(name),
            exist_ok=True,
            **hparams,
        )
        # best.pt 路径以 ultralytics 实际落盘为准（trainer.best），不手工拼，免受
        # 其 save_dir 解析规则影响。
        best = Path(model.trainer.best)
        num_params = sum(p.numel() for p in model.model.parameters())
        names = {int(k): v for k, v in dict(model.names).items()}
        return best, num_params, names, len(names)

    def val(
        self, weights, data_yaml, split: str, imgsz: int, device, *,
        conf: float, iou: float, max_det: int, agnostic_nms: bool,
    ) -> dict:
        """在指定 split 上验证，返回与 ultralytics 解耦的普通 dict。

        ``per_class`` 只含验证集里有样本、被评估到的类别（``ap_class_index``）；
        ``names`` 是 data.yaml 声明的全部类别 —— 二者的差集即"无样本类别"，
        由 ``build_detection_metrics`` 标为 MISSING。
        """
        from ultralytics import YOLO

        model = YOLO(str(weights))
        # Ultralytics validation 可能 fuse Conv/BN 并替换 model.model；必须在此之前统计，
        # 才能记录 checkpoint 原始结构的参数量，而不是优化后的推理结构。
        num_params = sum(parameter.numel() for parameter in model.model.parameters())
        m = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            device=_ul_device(device),
            conf=conf,
            iou=iou,
            max_det=max_det,
            agnostic_nms=agnostic_nms,
            verbose=False,
        )
        box = m.box
        names = {int(k): v for k, v in dict(model.names).items()}
        per_class = {}
        for i, cidx in enumerate(list(box.ap_class_index)):
            per_class[names[int(cidx)]] = {
                "precision": float(box.p[i]),
                "recall": float(box.r[i]),
                "map50": float(box.ap50[i]),
            }
        return {
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
            "num_params": num_params,
            "names": names,
            "per_class": per_class,
        }

    def predict(
        self, weights, data_yaml, split: str, imgsz: int, device, *,
        conf: float, iou: float, max_det: int, agnostic_nms: bool,
    ) -> dict:
        """逐图推理并返回原始检测事实，框使用归一化 ``xywh``。

        真值仍由钉定的 YOLO testset manifest/data.yaml 提供。该旁路与 ``val`` 分开，避免
        依赖 Ultralytics 内部 validator 状态，也不在适配器内决定 artifact schema。
        """
        import yaml
        from ultralytics import YOLO

        data_yaml = Path(data_yaml).resolve()
        payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        root = Path(str(payload.get("path") or data_yaml.parent)).expanduser()
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        configured = payload.get(split)
        if configured is None:
            raise ValueError(f"YOLO data.yaml 未声明 split={split}")
        sources = configured if isinstance(configured, list) else [configured]
        resolved_sources = []
        for value in sources:
            source = Path(str(value)).expanduser()
            resolved_sources.append(str(source if source.is_absolute() else (root / source).resolve()))

        model = YOLO(str(weights))
        items = {}
        source_arg = resolved_sources[0] if len(resolved_sources) == 1 else resolved_sources
        for result in model.predict(
            source=source_arg,
            imgsz=imgsz,
            device=_ul_device(device),
            conf=conf,
            iou=iou,
            max_det=max_det,
            agnostic_nms=agnostic_nms,
            stream=True,
            verbose=False,
        ):
            boxes = []
            if result.boxes is not None:
                xywhn = result.boxes.xywhn.detach().cpu().tolist()
                classes = result.boxes.cls.detach().cpu().tolist()
                confidences = result.boxes.conf.detach().cpu().tolist()
                boxes = [
                    {
                        "class_id": int(class_id),
                        "confidence": float(confidence),
                        "xywhn": [float(value) for value in coords],
                    }
                    for class_id, confidence, coords in zip(classes, confidences, xywhn)
                ]
            items[Path(result.path).name] = {"predictions": boxes}
        return {
            "split": split,
            "labels": {str(key): value for key, value in dict(model.names).items()},
            "items": items,
        }

_ADAPTERS = {
    YoloAdapter.model_type: YoloAdapter,
}


def get_adapter(model_type: str) -> YoloAdapter:
    if model_type not in _ADAPTERS:
        raise KeyError(f"未注册的检测适配器: {model_type}；已注册: {sorted(_ADAPTERS)}")
    return _ADAPTERS[model_type]()
