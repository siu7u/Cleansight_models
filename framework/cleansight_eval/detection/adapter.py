"""YOLO 检测适配器（封装 ultralytics，检测纵专属）。

检测纵不套用任何跨域 family Protocol：ultralytics 自持训练/验证，本适配器只暴露
``train`` / ``val`` 两个方法，由 ``get_adapter(family_id)`` 取用。这与 temporal 纵的
family 是**两套不相交的契约**——故意不强行统一。

本适配器同时充当**检测的 data loader**：检测的 features 契约就是**图像**、feeding
契约是 **single_frame**，ultralytics 从 ``data.yaml`` 一次性读入 images/labels 并自持
批处理——无需另写 loader。

ultralytics/torch 为重依赖，全部在方法内部 import，使仅做数据/纯逻辑的场景
（如检测指标单元测试、注入假 adapter 的冒烟）无需安装它们。
"""

from __future__ import annotations

from pathlib import Path


def _ul_device(device) -> str:
    """torch.device -> ultralytics 的 device 参数字符串。"""
    t = getattr(device, "type", None) or str(device)
    if t == "cuda":
        idx = getattr(device, "index", None)
        return str(idx) if idx is not None else "0"
    return t  # "mps" / "cpu"


class YoloAdapter:
    family_id = "yolo"

    def train(self, weights, data_yaml, train_cfg: dict, imgsz: int, device, project, name):
        """训练 YOLO，返回 (best_pt, num_params, names, nc)。

        ultralytics 自行把权重写到 ``project/name/weights/best.pt``；本方法不接管
        权重落盘，由 DetectionTask 另写 sidecar 元信息。
        """
        from ultralytics import YOLO

        model = YOLO(str(weights))
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
        )
        # best.pt 路径以 ultralytics 实际落盘为准（trainer.best），不手工拼，免受
        # 其 save_dir 解析规则影响。
        best = Path(model.trainer.best)
        num_params = sum(p.numel() for p in model.model.parameters())
        names = {int(k): v for k, v in dict(model.names).items()}
        return best, num_params, names, len(names)

    def val(self, weights, data_yaml, split: str, imgsz: int, device) -> dict:
        """在指定 split 上验证，返回与 ultralytics 解耦的普通 dict。

        ``per_class`` 只含验证集里有样本、被评估到的类别（``ap_class_index``）；
        ``names`` 是 data.yaml 声明的全部类别 —— 二者的差集即"无样本类别"，
        由 ``build_detection_metrics`` 标为 MISSING。
        """
        from ultralytics import YOLO

        model = YOLO(str(weights))
        m = model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            device=_ul_device(device),
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
            "names": names,
            "per_class": per_class,
        }


_ADAPTERS = {
    YoloAdapter.family_id: YoloAdapter,
}


def get_adapter(family_id: str) -> YoloAdapter:
    if family_id not in _ADAPTERS:
        raise KeyError(f"未注册的检测适配器: {family_id}；已注册: {sorted(_ADAPTERS)}")
    return _ADAPTERS[family_id]()
