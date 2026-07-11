"""特征来源溯源（需求 §4.4）。

描述特征来自哪里、由什么方法生成，以便未来把 YOLO 替换为其他检测器或视觉
编码器，而时序模型只依赖稳定的 feature schema。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureSource:
    method: str  # 例如 "yolo"
    version: str  # 例如上游 YOLO 权重版本
    notes: str = ""

    def to_dict(self) -> dict:
        return {"method": self.method, "version": self.version, "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSource":
        return cls(method=data.get("method", "unknown"), version=data.get("version", ""), notes=data.get("notes", ""))
