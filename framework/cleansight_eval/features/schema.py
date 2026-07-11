"""特征来源抽象：feature schema（需求 §4.4）。

时序模型只依赖稳定的 feature schema，不绑定具体 YOLO 实现。schema 描述特征
维度与版本，供训练前兼容检查（§7.3）。本次只定义 schema 与校验，不迁移在线
抽取器实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FeatureSchema:
    """模型期望的特征契约。

    ``dim``：特征维度；``version``：特征映射版本（如 ``legacy-20d-v1``）；
    ``layout``：可选的逐通道语义说明。
    """

    dim: int
    version: str
    layout: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"dim": self.dim, "version": self.version, "layout": self.layout}

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSchema":
        return cls(dim=data["dim"], version=data["version"], layout=data.get("layout", []))

    def validate_array(self, features: np.ndarray) -> None:
        """校验特征矩阵 ``[T, F]`` 的维度与 schema 一致。"""

        if features.ndim != 2:
            raise ValueError(f"特征应为二维 [T, F]，实际 {features.shape}")
        if features.shape[1] != self.dim:
            raise ValueError(
                f"特征维度与 schema 不一致: 实际={features.shape[1]} 期望={self.dim} "
                f"(version={self.version})"
            )

    def is_compatible(self, other: "FeatureSchema") -> bool:
        return self.dim == other.dim and self.version == other.version
