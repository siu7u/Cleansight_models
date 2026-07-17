"""训练 history.csv 写入工具。"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any


class HistoryWriter:
    """稳定追加逐 epoch 指标；文件不存在时自动写表头。"""

    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def append(self, row: dict[str, Any]) -> None:
        """按固定列追加一行；缺失字段写空字符串，便于 diff 和脚本读取。"""

        normalized = {name: row.get(name, "") for name in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(normalized)


def _read_numeric_history(path: Path) -> tuple[list[float], dict[str, list[float | None]]]:
    """读取 history.csv，把空值和非有限值保留为不可绘制点。"""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"训练 history 没有 epoch 记录: {path}")

    epochs: list[float] = []
    series: dict[str, list[float | None]] = {}
    for row_index, row in enumerate(rows, start=1):
        try:
            epoch = float(row.get("epoch") or row_index)
        except ValueError as exc:
            raise ValueError(f"history 第 {row_index} 行 epoch 非数值") from exc
        epochs.append(epoch)
        for name, raw_value in row.items():
            if name == "epoch":
                continue
            value: float | None = None
            if raw_value not in (None, ""):
                try:
                    candidate = float(raw_value)
                    value = candidate if math.isfinite(candidate) else None
                except ValueError:
                    pass
            series.setdefault(name, []).append(value)
    return epochs, series


def plot_training_history(history_path: str | Path, output_path: str | Path) -> Path:
    """把逐 epoch history 渲染为 loss、验证指标、学习率和耗时四面板 PNG。

    本函数只读取 ``history.csv``，不参与训练状态更新。matplotlib 使用无界面的 Agg
    backend，适用于服务器训练；输出父目录会自动创建。
    """

    history_path = Path(history_path)
    output_path = Path(output_path)
    epochs, series = _read_numeric_history(history_path)

    # 部署机的 HOME 可能只读；显式使用临时缓存，避免 matplotlib 导入时产生权限告警。
    cache_dir = Path(tempfile.gettempdir()) / "cleansight-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    def draw(axis, names: list[str], title: str, ylabel: str) -> None:
        drawn = False
        for name in names:
            values = series.get(name)
            if not values or not any(value is not None for value in values):
                continue
            axis.plot(epochs, values, marker="o", linewidth=1.6, markersize=3, label=name)
            drawn = True
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        if drawn:
            axis.legend()
        else:
            axis.text(0.5, 0.5, "no data", ha="center", va="center", transform=axis.transAxes)

    draw(axes[0, 0], ["train_loss", "val_loss"], "Loss", "loss")
    draw(axes[0, 1], ["val_acc", "val_edit", "val_f1_0.5"], "Validation metrics", "percent")
    draw(axes[1, 0], ["lr"], "Learning rate", "lr")
    draw(axes[1, 1], ["epoch_sec"], "Epoch duration", "seconds")
    figure.suptitle(f"Training history · {history_path.parent.name}")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def try_plot_training_history(
    history_path: str | Path,
    output_path: str | Path,
) -> tuple[Path | None, str | None]:
    """尽力生成训练曲线；绘图失败返回原因，不影响 checkpoint 和训练结果。"""

    try:
        return plot_training_history(history_path, output_path), None
    except Exception as exc:  # 绘图是旁路产物，依赖/文件错误不能拖垮训练
        return None, f"{type(exc).__name__}: {exc}"
