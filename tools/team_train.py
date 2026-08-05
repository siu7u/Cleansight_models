#!/usr/bin/env python3
"""
组员训练入口：一条命令训练任意已注册模型，无需手动挑选/编辑实验 YAML。

用法（仓库根执行）:
    # 列出所有可训练模型与示例命令（无需 torch）
    python tools/team_train.py --list

    # YOLO 指定规模与组（默认 yolo11n / 各自默认组）
    python tools/team_train.py --model yolo11s --group group1_large
    python tools/team_train.py --model yolo --group group1_large

    # 时序模型
    python tools/team_train.py --model gru
    python tools/team_train.py --model mstcn
    python tools/team_train.py --model mstcn2
    python tools/team_train.py --model transformer

    # ROI 特征融合（需指定目标类别）
    python tools/team_train.py --model feature_fusion -S data.classes=air_gun

    # 透传任意超参覆盖（同 framework cli.train 的 -S）
    python tools/team_train.py --model yolo11m --group group1_large \
        -S train.epochs=200 -S model.imgsz=960

    # 训练前跳过数据就绪检查（默认会自动检查并提示下载）
    python tools/team_train.py --model gru --force

训练前会自动检查所需数据集是否就绪（见 tools/team_dataset.py --check），
缺失时打印下载命令并退出；--force 跳过检查。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── 模型 → 实验配置映射（--list 与解析共用；不解析 yaml，保证输出稳定）──
# 每个条目: (模型名, 配置路径, 默认组, 所需数据集, 说明)
MODELS: dict[str, dict] = {
    "yolo": {
        "config": "framework/experiments/yolo-clean-large.yaml",
        "group": "group1_large",
        "dataset": "yolo",
        "desc": "YOLO 目标检测（默认 yolo11n，可用 --model yolo11s/m/l 换规模）",
    },
    "yolo11n": {"config": "framework/experiments/yolo-clean-large.yaml", "group": "group1_large",
                "dataset": "yolo", "weights": "yolo11n.pt", "desc": "YOLO11 nano 检测"},
    "yolo11s": {"config": "framework/experiments/yolo-clean-large.yaml", "group": "group1_large",
                "dataset": "yolo", "weights": "yolo11s.pt", "desc": "YOLO11 small 检测"},
    "yolo11m": {"config": "framework/experiments/yolo-clean-large.yaml", "group": "group1_large",
                "dataset": "yolo", "weights": "yolo11m.pt", "desc": "YOLO11 medium 检测"},
    "gru": {"config": "framework/experiments/gru-actionmixed.yaml", "group": None,
            "dataset": "actionmixed", "desc": "GRU 因果滑窗时序"},
    "mstcn": {"config": "framework/experiments/mstcn-actionmixed.yaml", "group": None,
              "dataset": "actionmixed", "desc": "MS-TCN 全序列时序"},
    "mstcn2": {"config": "framework/experiments/mstcn2-actionmixed.yaml", "group": None,
               "dataset": "actionmixed", "desc": "MS-TCN++ 全序列时序"},
    "transformer": {"config": "framework/experiments/transformer-actionmixed.yaml", "group": None,
                    "dataset": "actionmixed", "desc": "Transformer 全序列时序"},
    "feature_fusion": {"config": "framework/experiments/roi-fusion.yaml", "group": None,
                       "dataset": "yolo", "desc": "ROI 特征融合（需 -S data.classes=... 指定类别）"},
}

# YOLO 规模别名 → 权重文件
YOLO_WEIGHTS = {"yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt", "yolo11m": "yolo11m.pt"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="组员模型训练入口", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo", help="模型名（见 --list）")
    p.add_argument("--group", default=None, help="YOLO 分组（group1_large / group2_small），默认按模型")
    p.add_argument("--list", action="store_true", help="列出所有可训练模型与示例命令")
    p.add_argument("--force", action="store_true", help="跳过训练前数据就绪检查")
    p.add_argument("--runs-dir", default="runs", help="运行输出根目录（默认 runs）")
    p.add_argument("-S", "--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="通用配置覆盖，点路径寻址，可多次；如 -S train.epochs=200")
    return p.parse_args(argv)


def list_models() -> str:
    """生成模型清单文本（无 torch 依赖）。"""

    lines = ["可训练模型：", ""]
    for name, info in sorted(MODELS.items()):
        if name.startswith("yolo11"):
            continue  # yolo11n/s/m 归并在 yolo 条目下展示
        cmd = f"python tools/team_train.py --model {name}"
        if info["group"] is None and name == "feature_fusion":
            cmd += ' -S data.classes=<类名>'
        lines.append(f"  {name:<14} {info['desc']}")
        lines.append(f"      {cmd}")
        lines.append("")
    lines += [
        "YOLO 规模（同一配置，--group 选组）：",
        "  python tools/team_train.py --model yolo11n --group group1_large",
        "  python tools/team_train.py --model yolo11s --group group2_small",
        "  python tools/team_train.py --model yolo11m --group group1_large",
        "",
        "提示：训练前会自动检查数据；缺失时先运行 tools/team_dataset.py --preset all。",
    ]
    return "\n".join(lines)


def resolve_model(args) -> dict:
    """解析模型名 → (配置路径, group, weights, dataset)，未知模型报错。"""

    name = args.model
    if name not in MODELS:
        known = sorted({k for k in MODELS if not k.startswith("yolo11")} | {"yolo11n", "yolo11s", "yolo11m"})
        raise SystemExit(f"未知模型 '{name}'。可用: {', '.join(known)}（用 --list 查看详情）")

    info = dict(MODELS[name])
    if name in YOLO_WEIGHTS:
        info["weights"] = YOLO_WEIGHTS[name]
    if args.group:
        info["group"] = args.group
    return info


def _coerce(value: str):
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _parse_overrides(items: list[str]) -> list[tuple[str, object]]:
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set 需为 KEY=VALUE 形式（点路径寻址）: {item!r}")
        key, value = item.split("=", 1)
        out.append((key.strip(), _coerce(value.strip())))
    return out


def check_dataset(dataset_key: str) -> None:
    """调用 tools/team_dataset.py --check 的子集逻辑，缺失时打印下载命令并退出。"""

    sys.path.insert(0, str(ROOT))
    from tools.team_dataset import check_required_datasets

    missing = check_required_datasets([dataset_key])
    if missing:
        print("\n[team_train] 训练所需数据未就绪，请先下载：")
        for key in missing:
            print(f"  python tools/team_dataset.py --preset {key}")
        raise SystemExit(f"缺少数据集: {', '.join(missing)}（或用 --force 跳过检查）")


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        print(list_models())
        return 0

    info = resolve_model(args)
    config_path = ROOT / info["config"]
    if not config_path.is_file():
        raise SystemExit(f"配置文件不存在: {config_path}")

    # 数据就绪检查
    if not args.force:
        check_dataset(info["dataset"])

    # 组装覆盖项：YOLO 规模权重 + 组别 + 用户 -S
    overrides = _parse_overrides(args.overrides)
    if info.get("weights"):
        overrides.append(("model.weights", info["weights"]))
    if info.get("group"):
        overrides.append(("data.name", info["group"]))

    # 复用 framework 训练逻辑（延迟 import，保证 --list 无 torch 可用）
    from framework.cleansight_eval.cli import train as train_cli
    from framework.cleansight_eval.core.config import apply_overrides, load_config
    from framework.cleansight_eval.core.environment import pick_device
    from framework.cleansight_eval.core.registry import get_pipeline

    print(f"[team_train] 模型: {args.model}  配置: {info['config']}")
    if overrides:
        print(f"[team_train] 覆盖: {overrides}")

    cfg = apply_overrides(load_config(str(config_path)), overrides)
    device = pick_device()
    pipeline = get_pipeline(cfg["pipeline"])
    pipeline.validate_config(cfg)
    ckpt = pipeline.train(cfg, runs_dir=args.runs_dir, seed=42, device=device)
    print(f"[team_train] 完成，checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
