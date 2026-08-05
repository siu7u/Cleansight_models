#!/usr/bin/env python3
"""把 CleanSight ModelScope 数据集下载为原始仓库文件。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download


DATASET_ID = "lhh010/cleansight-ActionMixed"
DEFAULT_OUTPUT = Path("raw/modelscope/cleansight-ActionMixed")


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
    parser.add_argument("--dataset", default=DATASET_ID, help="ModelScope dataset id.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Local output directory for the dataset files.",
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


def main() -> None:
    args = parse_args()
    pipeline_root = Path(__file__).resolve().parent
    repo_root = pipeline_root.parents[1]
    load_env(repo_root / ".env")
    load_env(pipeline_root / ".env")

    token = os.environ.get(args.token_env)
    if not token:
        raise SystemExit(f"Missing token: set {args.token_env} or put it in .env")

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = pipeline_root / output
    output.mkdir(parents=True, exist_ok=True)

    HubApi().login(token)
    path = snapshot_download(
        repo_id=args.dataset,
        repo_type="dataset",
        local_dir=str(output),
        token=token,
        max_workers=args.workers,
    )
    print(f"Downloaded dataset files to: {path}")


if __name__ == "__main__":
    main()
