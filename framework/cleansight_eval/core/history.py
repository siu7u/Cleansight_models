"""训练 history.csv 写入工具。"""

from __future__ import annotations

import csv
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
