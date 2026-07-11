"""YOLO 检测模型族（封装 ultralytics）。

模型族契约按任务分化（§4.2）：时序族 ``ModelFamily`` 返回 ``[B,T,C]`` logits，
而 YOLO 的训练/验证由 ultralytics 封装，暴露的是 ``train`` / ``val``。二者都由
``get_family(family_id)`` 取用，由所属 Task 决定调用哪套方法。

同时这里也充当**检测任务的 data loader**：与时序任务"读原始数据+按 features/feeding
契约转成模型输入"对应，检测的 features 契约就是**图像**、feeding 契约是 **single_frame**，
ultralytics 从 ``data.yaml`` 一次性读入 images/labels 并自持批处理——无需另写 loader。

ultralytics/torch 为重依赖，全部在方法内部 import，使仅做数据/纯逻辑的场景
（如检测指标单元测试）无需安装它们。
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


class YoloFamily:
    family_id = "yolo"

    def train(self, weights, data_yaml, train_cfg: dict, imgsz: int, device, project, name):
        """训练 YOLO，返回 (best_pt, num_params, names, nc)。

        ultralytics 自行把权重写到 ``project/name/weights/best.pt``；本方法不接管
        权重落盘，由 DetectionTask 另写 sidecar 元信息。
        """
        from ultralytics import YOLO

        model = YOLO(str(weights))
        model.train(
            data=str(data_yaml),
            epochs=train_cfg.get("epochs", 100),
            imgsz=imgsz,
            batch=train_cfg.get("batch", 16),
            patience=train_cfg.get("patience", 20),
            device=_ul_device(device),
            project=str(project),
            name=str(name),
            exist_ok=True,
        )
        best = Path(project) / str(name) / "weights" / "best.pt"
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


def get_family() -> YoloFamily:
    return YoloFamily()
