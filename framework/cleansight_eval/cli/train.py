"""训练入口：python -m cleansight_eval.cli.train --config <yaml>。

配置驱动同架构变体。本入口只做**分派**：加载配置、选设备、应用 CLI 覆盖项，然后交给
``get_pipeline(cfg["pipeline"]).train(...)``。训练主体（时序的 forward/loss 循环、检测的
ultralytics 封装）各由所属流水线实现。
"""

from __future__ import annotations

import argparse

from ..core.config import apply_overrides, load_config
from ..core.environment import pick_device
from ._registry import get_pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="cleansight_eval 训练入口")
    p.add_argument("--config", required=True, help="实验配置 YAML")
    p.add_argument("--runs-dir", default="runs", help="运行输出根目录")
    p.add_argument("--seed", type=int, default=42)
    # 覆盖项（不改配置文件即可临时调参）
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--window", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None) -> str:
    args = parse_args(argv)
    cfg = apply_overrides(
        load_config(args.config),
        {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "window": args.window},
    )
    device = pick_device()
    pipeline = get_pipeline(cfg["pipeline"])
    pipeline.validate_config(cfg)  # 流水线专属校验（core 不再代劳）
    return pipeline.train(cfg, runs_dir=args.runs_dir, seed=args.seed, device=device)


if __name__ == "__main__":
    main()
