"""YOLO registry CARD.md helpers.

This module centralizes CARD path, registry naming, and training-history
formatting so pipeline scripts can record model metadata without owning the
Markdown layout.
"""
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from utils.common import ROOT


REGISTRY = ROOT.parent / "registry"


def registry_name(group: str) -> str:
    """Map a pipeline group name to its YOLO registry version directory."""

    return f"yolo-{group.replace('_', '-')}-v1"


def rel_to_yolo_root(path: Path) -> str:
    """Return a path relative to `yolo-detection/` when possible."""

    try:
        return str(path.relative_to(ROOT.parent))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class YoloTrainingRecord:
    """Metadata needed to append one YOLO training run to CARD.md."""

    group: str
    class_names: list[str]
    dataset: Path
    config: Path
    base_model: str
    epochs: int
    imgsz: int
    batch: int
    patience: int
    device: str
    run_weight: Path
    versioned_weight: Path
    timestamp: str

    @classmethod
    def from_training(
        cls,
        group: str,
        cfg: dict,
        device: str,
        run_weight: Path,
        versioned_weight: Path,
        timestamp: str | None = None,
    ) -> "YoloTrainingRecord":
        """Build a CARD record from the shared pipeline config and outputs."""

        tcfg = cfg.get("train", {})
        return cls(
            group=group,
            class_names=list(cfg.get("groups", {}).get(group, [])),
            dataset=ROOT / "datasets" / group / "data.yaml",
            config=ROOT / "config.yaml",
            base_model=tcfg.get("model", "yolo11n.pt"),
            epochs=tcfg.get("epochs", 100),
            imgsz=tcfg.get("imgsz", 640),
            batch=tcfg.get("batch", 16),
            patience=tcfg.get("patience", 20),
            device=device,
            run_weight=run_weight,
            versioned_weight=versioned_weight,
            timestamp=timestamp or datetime.now().strftime("%Y%m%d-%H%M%S"),
        )


class YoloCardWriter:
    """Append YOLO model metadata to registry CARD.md files."""

    def __init__(self, registry_root: Path = REGISTRY):
        self.registry_root = registry_root

    def card_path(self, group: str) -> Path:
        """Return the CARD.md path for a pipeline group."""

        return self.registry_root / registry_name(group) / "CARD.md"

    def append_training_history(self, record: YoloTrainingRecord) -> Path:
        """Append one training-history block and return the updated CARD path."""

        card = self.card_path(record.group)
        card.parent.mkdir(parents=True, exist_ok=True)
        text = card.read_text(encoding="utf-8") if card.exists() else f"# 模型卡：{registry_name(record.group)}\n"
        if "## 训练历史" not in text:
            text = text.rstrip() + "\n\n## 训练历史\n"
        text = text.rstrip() + "\n" + "\n".join(self._training_history_lines(record)) + "\n"
        card.write_text(text, encoding="utf-8")
        return card

    def _training_history_lines(self, record: YoloTrainingRecord) -> list[str]:
        """Render a training-history block for CARD.md."""

        return [
            "",
            f"### {record.timestamp}",
            "",
            f"- 分组: `{record.group}`",
            f"- 类别: `{', '.join(record.class_names)}`",
            f"- 数据集: `{rel_to_yolo_root(record.dataset)}`",
            f"- 配置文件: `{rel_to_yolo_root(record.config)}`",
            f"- 基础模型: `{record.base_model}`",
            f"- epochs: {record.epochs}",
            f"- imgsz: {record.imgsz}",
            f"- batch: {record.batch}",
            f"- patience: {record.patience}",
            f"- 设备: `{record.device}`",
            f"- 运行权重: `{rel_to_yolo_root(record.run_weight)}`",
            f"- 版本导出权重: `{rel_to_yolo_root(record.versioned_weight)}`",
        ]
