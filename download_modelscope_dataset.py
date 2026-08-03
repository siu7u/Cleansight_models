#!/usr/bin/env python3
"""把 CleanSight ModelScope 数据集下载为原始仓库文件。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

DATASET_PRESETS = {
    "actionmixed": {
        "dataset": "lhh010/cleansight-ActionMixed",
        "output": Path("datasets/raw/modelscope/cleansight-ActionMixed"),
        "description": "已打包的 ActionMixed 检测/时序样本",
    },
    "raw": {
        "dataset": "lhh010/cleansight-raw",
        "output": Path("datasets/raw/modelscope/cleansight-raw"),
        "description": "ModelScope cleansight-raw 原始数据集",
    },
}
DEFAULT_PRESET = "actionmixed"


def load_env(path: Path) -> None:
    """从本地 .env 读取简单 KEY=VALUE,且不打印密钥。"""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把 ModelScope 数据集仓库下载到本地目录。"
    )
    parser.add_argument(
        "--preset",
        choices=sorted(DATASET_PRESETS),
        default=DEFAULT_PRESET,
        help="预置数据源；默认 actionmixed。raw 对应 https://www.modelscope.cn/datasets/lhh010/cleansight-raw",
    )
    parser.add_argument(
        "--dataset",
        help="手动指定 ModelScope dataset id 或 URL；传入后覆盖 --preset 的 dataset。",
    )
    parser.add_argument(
        "--output",
        help="Local output directory for the dataset files. 不传则按 --preset 选择默认目录。",
    )
    parser.add_argument(
        "--token-env",
        default="MODELSCOPE_TOKEN",
        help="Environment variable that stores the ModelScope token.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel download workers. Lower this if ModelScope returns 400/429.",
    )
    return parser.parse_args()


def dataset_id_from_value(value: str) -> str:
    """接受 ``namespace/name`` 或 ModelScope dataset URL，统一转成 dataset id。"""

    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        parts = [part for part in parsed.path.split("/") if part]
        if "datasets" in parts:
            idx = parts.index("datasets")
            if len(parts) >= idx + 3:
                return f"{parts[idx + 1]}/{parts[idx + 2]}"
        raise ValueError(f"无法从 ModelScope URL 解析 dataset id: {value}")
    return value


def print_temporal_usage(output: Path) -> None:
    """下载 ActionMixed 后提示 framework 时序训练可直接使用的配置与命令。"""

    labels = output / "labels" / "data.yaml"
    frames = output / "frames" / "data.yaml"
    if not labels.exists() or not frames.exists():
        return

    print("Temporal training root:", output)
    print("Detected labels/data.yaml and frames/data.yaml; framework temporal configs can use this dataset.")
    print("Example:")
    print("  cd framework")
    print("  python -m cleansight_eval.cli.train --config experiments/gru-actionmixed.yaml")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    load_env(repo_root / ".env")

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"Missing token: set {args.token_env} or put it in .env")

    preset = DATASET_PRESETS[args.preset]
    dataset = dataset_id_from_value(args.dataset) if args.dataset else preset["dataset"]
    output = Path(args.output or preset["output"]).expanduser()
    if not output.is_absolute():
        output = repo_root / output
    output.mkdir(parents=True, exist_ok=True)

    from modelscope.hub.api import HubApi
    from modelscope.hub.snapshot_download import snapshot_download

    HubApi().login(token)
    path = snapshot_download(
        repo_id=dataset,
        repo_type="dataset",
        local_dir=str(output),
        token=token,
        max_workers=args.workers,
    )
    print(f"Dataset: {dataset}")
    print(f"Downloaded dataset files to: {path}")
    print_temporal_usage(output)


if __name__ == "__main__":
    main()
