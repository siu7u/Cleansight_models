"""训练入口：python -m framework.cleansight_eval.cli.train --config <yaml>。

配置驱动同架构变体。本入口只做**分派**：加载配置、选设备、应用 CLI 覆盖项，然后交给
``get_pipeline(cfg["pipeline"]).train(...)``。训练主体（时序的 forward/loss 循环、检测的
ultralytics 封装）各由所属流水线实现。

临时调参用通用的 ``-S/--set 点路径=值``（可多次），不改配置文件；核心 CLI 不预设任何纵的
调参名，各纵按自己的超参词汇寻址，如 ``-S train.epochs=5``（两纵通用）、``-S train.batch=8``
（检测/ultralytics）、``-S train.window=32``（时序滑窗）。
"""

from __future__ import annotations

import argparse
from typing import Any

from ..core.config import apply_overrides, load_config
from ..core.environment import pick_device
from ._registry import get_pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="cleansight_eval 训练入口")
    p.add_argument("--config", required=True, help="实验配置 YAML")
    p.add_argument("--runs-dir", default="runs", help="运行输出根目录")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", help="从完整训练 checkpoint（通常是 checkpoints/last.pt）恢复")
    p.add_argument(
        "-S",
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="通用配置覆盖，点路径寻址，可多次；如 -S train.epochs=5 -S train.batch=8",
    )
    return p.parse_args(argv)


def _coerce(s: str) -> Any:
    """把 --set 的字符串值转成 bool/None/int/float，无法转则原样保留字符串。"""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _parse_overrides(items: list[str]) -> list[tuple[str, Any]]:
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set 需为 KEY=VALUE 形式（点路径寻址）: {item!r}")
        key, value = item.split("=", 1)
        out.append((key.strip(), _coerce(value.strip())))
    return out


def main(argv=None) -> str:
    args = parse_args(argv)
    overrides = _parse_overrides(args.overrides)
    if args.resume:
        overrides.append(("train.resume", args.resume))
    cfg = apply_overrides(load_config(args.config), overrides)
    device = pick_device()
    pipeline = get_pipeline(cfg["pipeline"])
    pipeline.validate_config(cfg)  # 流水线专属校验（core 不再代劳）
    return pipeline.train(cfg, runs_dir=args.runs_dir, seed=args.seed, device=device)


if __name__ == "__main__":
    main()
