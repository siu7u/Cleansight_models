"""模型别名注册表：把短模型名解析为实验配置与覆盖项（framework 层）。

组员/脚本用 ``python -m framework.cleansight_eval.cli.train --model <名>`` 训练，
本模块负责把 ``yolo11s``、``gru`` 等别名解析成：
  - 实验配置路径（framework/experiments/*.yaml）
  - 覆盖项（YOLO 权重、分组、数据键），由 CLI 应用后走统一训练入口。

只做纯解析、不 import 重依赖（--list-models 无 torch 也可用）。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = REPO_ROOT / "framework" / "experiments"

# 模型名 → (配置文件名, 默认 group, 数据集键, 覆盖项, 说明)
# 数据集键对应 core/dataset_download.py 的 REQUIRED_FILES。
MODELS: dict[str, dict] = {
    "yolo": {
        "config": "yolo-clean-large.yaml",
        "group": "group1_large",
        "dataset": "yolo",
        "desc": "YOLO 目标检测（默认 yolo11n；规模用 yolo11n/s/m 别名）",
    },
    "yolo11n": {"config": "yolo-clean-large.yaml", "group": "group1_large", "dataset": "yolo",
                "overrides": {"model.weights": "yolo11n.pt"}, "desc": "YOLO11 nano"},
    "yolo11s": {"config": "yolo-clean-large.yaml", "group": "group1_large", "dataset": "yolo",
                "overrides": {"model.weights": "yolo11s.pt"}, "desc": "YOLO11 small"},
    "yolo11m": {"config": "yolo-clean-large.yaml", "group": "group1_large", "dataset": "yolo",
                "overrides": {"model.weights": "yolo11m.pt"}, "desc": "YOLO11 medium"},
    "gru": {"config": "gru-actionmixed.yaml", "group": None, "dataset": "actionmixed",
            "desc": "GRU 因果滑窗时序"},
    "mstcn": {"config": "mstcn-actionmixed.yaml", "group": None, "dataset": "actionmixed",
              "desc": "MS-TCN 全序列时序"},
    "mstcn2": {"config": "mstcn2-actionmixed.yaml", "group": None, "dataset": "actionmixed",
               "desc": "MS-TCN++ 全序列时序"},
    "transformer": {"config": "transformer-actionmixed.yaml", "group": None, "dataset": "actionmixed",
                    "desc": "Transformer 全序列时序"},
    "feature_fusion": {"config": "roi-fusion.yaml", "group": None, "dataset": "yolo",
                       "desc": "ROI 特征融合（需 -S data.classes=<类名>）"},
}

# 供 --list-models 展示的规模别名
SCALE_ALIASES = ("yolo11n", "yolo11s", "yolo11m")


def resolve_model(name: str, group: str | None = None) -> dict:
    """解析模型名 → 配置信息 dict；未知模型抛 SystemExit。"""

    if name not in MODELS:
        known = sorted({k for k in MODELS if not k.startswith("yolo11")} | set(SCALE_ALIASES))
        raise SystemExit(f"未知模型 '{name}'。可用: {', '.join(known)}（用 --list-models 查看详情）")

    info = dict(MODELS[name])
    if group:
        info["group"] = group
    return info


def model_config_path(info: dict) -> Path:
    """配置信息 → 实验配置绝对路径。"""

    return EXPERIMENTS / info["config"]


def list_models() -> str:
    """生成模型清单文本（无重依赖）。"""

    lines = ["可训练模型（python -m framework.cleansight_eval.cli.train --model <名>）：", ""]
    for name in sorted(k for k in MODELS if not k.startswith("yolo11")):
        info = MODELS[name]
        lines.append(f"  {name:<14} {info['desc']}")
        lines.append("")
    lines += [
        "YOLO 规模别名（同一配置，--group 选组）：",
        "  --model yolo11n --group group1_large",
        "  --model yolo11s --group group2_small",
        "  --model yolo11m --group group1_large",
        "",
        "提示：训练前会自动检查数据；缺失时先运行",
        "  python -m framework.cleansight_eval.cli.dataset --preset all",
    ]
    return "\n".join(lines)
