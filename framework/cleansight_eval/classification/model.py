"""ROI 图像分类模型（特征融合）。

对小目标/稀有类无法用 YOLO 检测框解决的问题，改用"ROI 区域分类"：预训练 CNN backbone
提取特征 + 轻量 MLP 多标签头，判断裁剪区域是否存在各类小目标。训练数据由
``data.build_roi_dataset`` 从 YOLO GT 框裁剪生成。
"""

from __future__ import annotations

from pathlib import Path

BACKBONE_CONFIGS = {
    "resnet18": {"input_size": 224, "feat_dim": 512, "pretrained": True},
    "resnet34": {"input_size": 224, "feat_dim": 512, "pretrained": True},
    "resnet50": {"input_size": 224, "feat_dim": 2048, "pretrained": True},
    "efficientnet-b0": {"input_size": 224, "feat_dim": 1280, "pretrained": True},
    "efficientnet-b1": {"input_size": 240, "feat_dim": 1280, "pretrained": True},
    "efficientnet-b2": {"input_size": 260, "feat_dim": 1408, "pretrained": True},
    "mobilenet-v3-small": {"input_size": 224, "feat_dim": 576, "pretrained": True},
}


class FeatureFusionModel:
    """
    图像特征融合模型:
      backbone (CNN) → Global Pool → Classifier (MLP) → Sigmoid (多标签)
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "resnet50",
        freeze_backbone: bool = False,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        import torch
        import torch.nn as nn
        import torchvision.models as tv_models

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        cfg = BACKBONE_CONFIGS[backbone_name]
        feat_dim = cfg["feat_dim"]
        self.input_size = cfg["input_size"]

        if backbone_name.startswith("resnet"):
            model_fn = getattr(tv_models, backbone_name)
            weights = getattr(tv_models, f"ResNet{backbone_name[6:]}_Weights").DEFAULT if cfg["pretrained"] else None
            backbone = model_fn(weights=weights)
            backbone.fc = nn.Identity()
        elif backbone_name.startswith("efficientnet"):
            model_fn = getattr(tv_models, backbone_name.replace("-", "_"))
            key = f"EfficientNet_{backbone_name.split('-')[1].upper()}_Weights"
            weights = getattr(tv_models, key).DEFAULT if cfg["pretrained"] else None
            backbone = model_fn(weights=weights)
            backbone.classifier = nn.Identity()
        elif backbone_name.startswith("mobilenet"):
            model_fn = getattr(tv_models, backbone_name.replace("-", "_"))
            key = f"MobileNet_V3_{'Small' if 'small' in backbone_name else 'Large'}_Weights"
            weights = getattr(tv_models, key).DEFAULT if cfg["pretrained"] else None
            backbone = model_fn(weights=weights)
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"不支持的 backbone: {backbone_name}")

        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_device(self, device_str: str = "auto") -> None:
        import torch

        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        self.backbone.to(self.device)
        self.classifier.to(self.device)

    def forward_from_features(self, features):
        return self.classifier(features)

    def state_dict(self):
        return {
            "backbone_state": self.backbone.state_dict(),
            "classifier_state": self.classifier.state_dict(),
        }

    def load_state_dict(self, state: dict, strict: bool = True):
        self.backbone.load_state_dict(state["backbone_state"], strict=strict)
        self.classifier.load_state_dict(state["classifier_state"], strict=strict)

    def save(self, path: Path) -> Path:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    @classmethod
    def load(cls, path: Path, num_classes: int, backbone_name: str = "resnet50") -> "FeatureFusionModel":
        import torch

        model = cls(num_classes=num_classes, backbone_name=backbone_name)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model
