#!/usr/bin/env python3
"""把 CleanSight ModelScope 数据集通过 git clone + git-lfs 拉取到本地 datasets/ 目录。

等效于官方命令::

    git lfs install
    git clone https://www.modelscope.cn/datasets/{namespace}/{name}.git

需要认证时，通过 ``GIT_LFS_SKIP_SMUDGE`` 环境变量可跳过 LFS 大文件下载。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

DATASET_PRESETS = {
    "actionmixed": {
        "dataset": "lhh010/cleansight-ActionMixed",
        "output": Path("datasets/cleansight-ActionMixed"),
        "description": "已打包的 ActionMixed 检测/时序样本",
    },
    "raw": {
        "dataset": "lhh010/cleansight-raw",
        "output": Path("datasets/cleansight-raw"),
        "description": "ModelScope cleansight-raw 原始数据集",
    },
    "yolo": {
        "dataset": "lhh010/cleansight-yolo",
        "output": Path("datasets/cleansight-yolo"),
        "description": "标准 YOLO 格式分组数据集（group1_large + group2_small，含 train/val/test）",
    },
}
DEFAULT_PRESET = "actionmixed"

MODELSCOPE_GIT_BASE = "https://www.modelscope.cn/datasets"


def load_env(path: Path) -> None:
    """从本地 .env 读取简单 KEY=VALUE，且不打印密钥。"""
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
        description="通过 git clone + git-lfs 把 ModelScope 数据集下载到 datasets/ 目录。"
    )
    parser.add_argument(
        "--preset",
        choices=sorted(DATASET_PRESETS),
        default=DEFAULT_PRESET,
        help=f"预置数据源（默认 {DEFAULT_PRESET}）",
    )
    parser.add_argument(
        "--dataset",
        help="手动指定 ModelScope dataset id（如 lhh010/cleansight-yolo）；传入后覆盖 --preset。",
    )
    parser.add_argument(
        "--output",
        help="本地输出目录；不传则按 --preset 选择默认目录（均在 datasets/ 下）。",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="git clone 的 --depth 参数（默认 1，即浅克隆）；设为 0 获取完整历史。",
    )
    parser.add_argument(
        "--branch",
        default="master",
        help="要克隆的分支名（默认 master）。",
    )
    parser.add_argument(
        "--skip-lfs",
        action="store_true",
        help="跳过 LFS 大文件下载（设置 GIT_LFS_SKIP_SMUDGE=1），"
        "仅获取文件指针。后续可手动 git lfs pull。",
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


def build_clone_url(dataset_id: str) -> str:
    """构造 ModelScope 官方 git clone URL。

    格式: https://www.modelscope.cn/datasets/{namespace}/{name}.git
    """
    return f"{MODELSCOPE_GIT_BASE}/{dataset_id}.git"


def check_git_lfs() -> bool:
    """检查 git-lfs 是否已安装并可执行。"""
    try:
        result = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"git-lfs: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return False


def ensure_git_lfs() -> None:
    """确保 git-lfs 已安装并初始化。未安装时给出安装提示。"""
    if check_git_lfs():
        # 初始化 git-lfs（幂等操作）
        subprocess.run(
            ["git", "lfs", "install"],
            capture_output=True, text=True,
        )
        return

    print("⚠️  未检测到 git-lfs，大文件（图片/视频）将无法正常下载。")
    print()
    print("请先安装 git-lfs：")
    print("  Linux:   sudo apt install git-lfs")
    print("  macOS:   brew install git-lfs")
    print("  Windows: 下载 https://git-lfs.com/")
    print()
    print("安装后重新运行本脚本即可。")
    print("或者使用 --skip-lfs 跳过 LFS 文件（仅获取文件指针），后续手动 git lfs pull。")
    raise SystemExit(1)


def git_clone(
    url: str,
    output: Path,
    branch: str,
    depth: int,
    skip_lfs: bool,
) -> None:
    """执行 git clone。

    如果目标目录已存在且是 git 仓库，则执行 git pull 增量更新。
    否则执行完整 clone。
    """
    if (output / ".git").is_dir():
        print(f"目标目录 {output} 已是 git 仓库，执行 git pull 更新...")
        _git_pull(output, branch, skip_lfs)
        return

    # 构建 clone 命令
    cmd = ["git", "clone"]
    if depth > 0:
        cmd.extend(["--depth", str(depth)])
    cmd.extend(["--branch", branch, "--single-branch"])
    cmd.extend([url, str(output)])

    print(f"git clone {'--depth ' + str(depth) + ' ' if depth > 0 else ''}"
          f"--branch {branch} {url}")
    print(f"      → {output}")
    print()

    env = os.environ.copy()
    if skip_lfs:
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        print("(GIT_LFS_SKIP_SMUDGE=1，跳过 LFS 文件下载)")

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        # 隐藏可能的敏感信息
        safe_stderr = result.stderr
        raise RuntimeError(
            f"git clone 失败 (exit {result.returncode}):\n{safe_stderr}\n"
            f"提示: 公开数据集不需要认证；"
            f"私有数据集请先配置 git credential 或 SSH key。"
        )

    print(f"克隆完成: {output}")
    _print_lfs_status(output)


def _git_pull(output: Path, branch: str, skip_lfs: bool) -> None:
    """对已有仓库执行 git pull 更新。"""
    env = os.environ.copy()
    if skip_lfs:
        env["GIT_LFS_SKIP_SMUDGE"] = "1"

    # 先 fetch
    result = subprocess.run(
        ["git", "-C", str(output), "fetch", "--depth", "1", "origin", branch],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        print(f"  git fetch 失败:\n{result.stderr}")
        print(f"  将删除目录重新克隆...")
        shutil.rmtree(output)
        return

    # merge
    result = subprocess.run(
        ["git", "-C", str(output), "merge", "--ff-only", f"origin/{branch}"],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        print(f"  git merge 失败:\n{result.stderr}")
        print(f"  将删除目录重新克隆...")
        shutil.rmtree(output)
        return

    print(f"  更新完成: {output}")
    _print_lfs_status(output)


def _print_lfs_status(output: Path) -> None:
    """打印 LFS 文件状态摘要。"""
    result = subprocess.run(
        ["git", "-C", str(output), "lfs", "ls-files", "--all"],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        lfs_files = [line.split()[-1] for line in result.stdout.strip().splitlines() if line.strip()]
        print(f"  LFS 文件: {len(lfs_files)} 个")
    elif result.returncode != 0:
        # 没有 LFS 文件或命令不支持
        pass


def print_temporal_usage(output: Path) -> None:
    """下载 ActionMixed 后提示 framework 时序训练可直接使用的配置与命令。"""

    labels = output / "labels" / "data.yaml"
    frames = output / "frames" / "data.yaml"
    if not labels.exists() or not frames.exists():
        return

    print()
    print("--- 时序训练 ---")
    print(f"  labels: {labels}")
    print(f"  frames: {frames}")
    print("  Example:")
    print("    python -m framework.cleansight_eval.cli.train \\")
    print("      --config framework/experiments/gru-actionmixed.yaml \\")
    print(f"      -S data.data_yaml={labels}")


def print_yolo_usage(output: Path) -> None:
    """下载 cleansight-yolo 后提示可直接使用的 data.yaml 与示例命令。"""

    data_yamls = sorted(output.glob("*/data.yaml"))
    if not data_yamls:
        return

    print()
    print("--- YOLO 训练 ---")
    for data_yaml in data_yamls:
        print(f"  data.yaml: {data_yaml}")
    print("  Example (framework):")
    print("    python -m framework.cleansight_eval.cli.train \\")
    print("      --config framework/experiments/yolo-clean-large.yaml \\")
    print(f"      -S data.data_yaml={data_yamls[0]}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    load_env(repo_root / ".env")

    preset = DATASET_PRESETS[args.preset]
    dataset_id = dataset_id_from_value(args.dataset) if args.dataset else preset["dataset"]
    output = Path(args.output or preset["output"]).expanduser()
    if not output.is_absolute():
        output = repo_root / output

    # 确保 git-lfs 可用
    ensure_git_lfs()

    clone_url = build_clone_url(dataset_id)
    print(f"数据源:     {dataset_id}")
    print(f"Clone URL:  {clone_url}")
    print(f"输出目录:   {output}")
    print()

    git_clone(clone_url, output, args.branch, args.depth, args.skip_lfs)

    if args.preset == "yolo":
        print_yolo_usage(output)
    else:
        print_temporal_usage(output)


if __name__ == "__main__":
    main()
