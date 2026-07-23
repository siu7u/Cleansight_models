"""以只追加方式维护旧模型 CARD 的训练与评估记录。

该工具只服务仍需复现的 legacy benchmark，不承担模型注册、训练调度或发布决策。
新 framework run 使用自身的 ``history.csv`` 和评测报告，不依赖本模块。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


FIELD_LABELS = {
    "run_id": "运行 ID",
    "timestamp": "时间",
    "model": "模型",
    "model_id": "模型 ID",
    "model_version": "模型版本",
    "mode": "模式",
    "training_mode": "训练模式",
    "dataset": "数据集",
    "dataset_path": "数据集路径",
    "dataset_version": "数据集版本",
    "split": "数据切分",
    "split_sha256": "切分 SHA256",
    "checkpoint": "权重",
    "checkpoint_path": "权重路径",
    "checkpoint_sha256": "权重 SHA256",
    "feature_mapping": "特征映射",
    "feature_mapping_version": "特征映射版本",
    "input_dim": "输入维度",
    "window": "窗口长度",
    "labels": "标签映射",
    "epochs": "训练轮数",
    "batch_size": "批大小",
    "learning_rate": "学习率",
    "lr": "学习率",
    "seed": "随机种子",
    "device": "设备",
    "inference_mode": "推理模式",
    "metrics": "评估指标",
    "report": "评估报告",
    "status": "状态",
    "command": "命令",
}


def file_sha256(path: str | Path) -> str:
    """分块计算文件 SHA256，并返回小写十六进制摘要。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_marker(section: str, run_id: str) -> str:
    """生成 CARD 记录的稳定去重标记。"""

    for name, value in (("section", section), ("run_id", run_id)):
        if not value or any(token in value for token in ("\r", "\n", "-->")):
            raise ValueError(f"{name} 不能为空，也不能包含换行或 '-->'")
    return f"<!-- cleansight-record:{section}:{run_id} -->"


def _field_label(key: Any) -> str:
    """把常用英文字段名转换为 CARD 使用的中文标签。"""

    text = str(key)
    return FIELD_LABELS.get(
        text,
        text
        if any("\u4e00" <= char <= "\u9fff" for char in text)
        else text.replace("_", " "),
    )


def _field_value(value: Any) -> str:
    """稳定渲染字段值，不解析或绝对化其中的相对路径。"""

    if value is None:
        return "未记录"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, separators=(", ", ": ")
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return ", ".join(_field_value(item) for item in value)
    return str(value)


def _last_level_two_heading(text: str) -> str | None:
    """返回 Markdown 中最后一个二级标题，便于在文件尾恢复目标章节。"""

    headings = [line[3:].strip() for line in text.splitlines() if line.startswith("## ")]
    return headings[-1] if headings else None


def append_card_record(
    card_path: str | Path,
    section: str,
    run_id: str,
    fields: Mapping[str, Any],
) -> bool:
    """在 CARD 文件尾追加记录，并按 ``(section, run_id)`` 去重。

    已有文件只会以二进制追加模式写入，原有前缀字节保持不变。返回 ``True`` 表示
    已追加，重复记录返回 ``False``。
    """

    if not isinstance(fields, Mapping):
        raise TypeError("fields 必须是映射类型")

    path = Path(card_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = _record_marker(str(section), str(run_id))

    with path.open("a+b") as stream:
        stream.seek(0)
        existing = stream.read()
        marker_bytes = marker.encode("utf-8")
        if marker_bytes in existing:
            return False

        text = existing.decode("utf-8") if existing else ""
        lines: list[str] = []
        if _last_level_two_heading(text) != section:
            lines.extend([f"## {section}", ""])
        lines.extend([marker, f"### {run_id}", "", f"- 运行 ID: `{run_id}`"])
        for key, value in fields.items():
            if str(key) == "run_id":
                continue
            lines.append(f"- {_field_label(key)}: `{_field_value(value)}`")

        prefix = "" if not existing or existing.endswith(b"\n") else "\n"
        block = (prefix + "\n".join(lines) + "\n").encode("utf-8")
        stream.seek(0, 2)
        stream.write(block)
        stream.flush()
    return True


def _record_parts(
    run_id_or_fields: str | Mapping[str, Any],
    fields: Mapping[str, Any] | None,
) -> tuple[str, Mapping[str, Any]]:
    """兼容独立 run_id 参数和包含 run_id 的完整记录字典。"""

    if isinstance(run_id_or_fields, Mapping):
        if fields is not None:
            raise TypeError("传入完整记录字典时不能再传 fields")
        record = dict(run_id_or_fields)
        run_id = record.pop("run_id", None)
        if run_id is None:
            raise ValueError("记录字典必须包含 run_id")
        return str(run_id), record
    if fields is None:
        raise TypeError("独立传入 run_id 时必须同时传 fields")
    return str(run_id_or_fields), fields


def append_training_record(
    card_path: str | Path,
    run_id_or_fields: str | Mapping[str, Any],
    fields: Mapping[str, Any] | None = None,
) -> bool:
    """把训练记录追加到“训练历史”，可直接传入含 run_id 的字典。"""

    run_id, record = _record_parts(run_id_or_fields, fields)
    return append_card_record(card_path, "训练历史", run_id, record)


def append_evaluation_record(
    card_path: str | Path,
    run_id_or_fields: str | Mapping[str, Any],
    fields: Mapping[str, Any] | None = None,
) -> bool:
    """把评估记录追加到“评估历史”，可直接传入含 run_id 的字典。"""

    run_id, record = _record_parts(run_id_or_fields, fields)
    return append_card_record(card_path, "评估历史", run_id, record)
