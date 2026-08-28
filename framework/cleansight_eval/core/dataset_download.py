"""数据集下载与就绪校验（framework 数据契约层）。

数据契约（catalog）只负责"登记与校验"；本模块负责把契约登记的数据集
从 ModelScope 下载到本地 `datasets/` 并在训练前校验就绪。CLI 入口见
``cli/dataset.py``（``python -m framework.cleansight_eval.cli.dataset``）。

下载逻辑从仓库根 `download_modelscope_dataset.py` 迁移而来，保持同一行为；
根脚本保留为薄封装（向后兼容）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELSCOPE_GIT_BASE = "https://www.modelscope.cn/datasets"

# 训练所需数据集 → 校验文件（相对仓库根）与下载 preset 键
REQUIRED_FILES: dict[str, list[Path]] = {
    "yolo": [
        Path("datasets/cleansight-yolo/group1_large/data.yaml"),
        Path("datasets/cleansight-yolo/group2_small/data.yaml"),
    ],
    "actionmixed": [
        Path("datasets/cleansight-ActionMixed/labels/data.yaml"),
        Path("datasets/cleansight-ActionMixed/frames/data.yaml"),
    ],
    "actionmixed-auto": [
        Path("datasets/cleansight-ActionMixed-auto/labels/data.yaml"),
        Path("datasets/cleansight-ActionMixed-auto/frames/data.yaml"),
        Path("datasets/cleansight-ActionMixed-auto/task_ids.yaml"),
    ],
}

DATASET_PRESETS: dict[str, dict] = {
    "actionmixed": {
        "dataset": "lhh010/cleansight-ActionMixed",
        "output": Path("datasets/cleansight-ActionMixed"),
        "description": "已打包的 ActionMixed 检测/时序样本",
    },
    "actionmixed-auto": {
        "dataset": "lhh010/cleansight-ActionMixed-auto",
        "output": Path("datasets/cleansight-ActionMixed-auto"),
        "description": "自动标注 ActionMixed 时序样本（YOLO 检测框 + 人工动作标签，v3）",
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


def check_required_datasets(keys: list[str] | None = None, root: Path = REPO_ROOT) -> list[str]:
    """校验指定数据集（默认全部）是否就绪，返回缺失的数据集 key 列表。"""

    keys = keys or list(REQUIRED_FILES)
    return [key for key in keys if not all((root / rel).is_file() for rel in REQUIRED_FILES[key])]


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
    return f"{MODELSCOPE_GIT_BASE}/{dataset_id}.git"


def check_git_lfs() -> bool:
    try:
        result = subprocess.run(["git", "lfs", "version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, Exception):
        return False


def ensure_git_lfs() -> None:
    if check_git_lfs():
        subprocess.run(["git", "lfs", "install"], capture_output=True, text=True)
        return
    print("⚠️  未检测到 git-lfs，大文件（图片/视频）将无法正常下载。")
    print("请先安装 git-lfs：")
    print("  Linux:   sudo apt install git-lfs")
    print("  macOS:   brew install git-lfs")
    print("  Windows: 下载 https://git-lfs.com/")
    print("或者使用 --skip-lfs 跳过 LFS 文件（仅获取文件指针）。")
    raise SystemExit(1)


def git_clone(url: str, output: Path, branch: str, depth: int, skip_lfs: bool) -> None:
    if (output / ".git").is_dir():
        print(f"目标目录 {output} 已是 git 仓库，执行 git pull 更新...")
        _git_pull(output, branch, skip_lfs)
        return

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
        raise RuntimeError(
            f"git clone 失败 (exit {result.returncode}):\n{result.stderr}\n"
            f"提示: 公开数据集不需要认证；私有数据集请先配置 git credential 或 SSH key。"
        )
    print(f"克隆完成: {output}")
    _print_lfs_status(output)


def _git_pull(output: Path, branch: str, skip_lfs: bool) -> None:
    env = os.environ.copy()
    if skip_lfs:
        env["GIT_LFS_SKIP_SMUDGE"] = "1"

    result = subprocess.run(
        ["git", "-C", str(output), "fetch", "--depth", "1", "origin", branch],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        print(f"  git fetch 失败:\n{result.stderr}")
        print(f"  将删除目录重新克隆...")
        shutil.rmtree(output)
        return

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
    result = subprocess.run(
        ["git", "-C", str(output), "lfs", "ls-files", "--all"], capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        lfs_files = [line.split()[-1] for line in result.stdout.strip().splitlines() if line.strip()]
        print(f"  LFS 文件: {len(lfs_files)} 个")


def list_presets() -> str:
    """打印全部预置数据源与目标位置。"""

    lines = ["可用预置数据源：", ""]
    for key, preset in sorted(DATASET_PRESETS.items()):
        lines.append(f"  {key:<12} {preset['description']}")
        lines.append(f"              → {preset['output']}")
    lines += [
        "",
        "  all          依次下载训练所需的全部数据集（yolo + actionmixed）",
        "",
        "示例：",
        "  python -m framework.cleansight_eval.cli.dataset --preset all",
        "  python -m framework.cleansight_eval.cli.dataset --check",
    ]
    return "\n".join(lines)


def check_data(root: Path = REPO_ROOT) -> list[str]:
    """校验已下载数据，返回缺失项 [(数据集key, 下载命令)]。"""

    missing = []
    print("数据就绪检查：")
    for key, files in REQUIRED_FILES.items():
        ok = all((root / rel).is_file() for rel in files)
        print(f"  [{'✅' if ok else '❌'}] {key}: {', '.join(str(f) for f in files)}")
        if not ok:
            missing.append((key, f"python -m framework.cleansight_eval.cli.dataset --preset {key}"))
    print()
    return missing


def download_one(preset_key: str, *, root: Path = REPO_ROOT, dataset: str | None = None,
                 output: str | None = None, branch: str = "master", depth: int = 1,
                 skip_lfs: bool = False) -> None:
    """下载单个 preset 到 datasets/ 正确位置。"""

    preset = DATASET_PRESETS[preset_key]
    dataset_id = dataset_id_from_value(dataset) if dataset else preset["dataset"]
    out = Path(output or preset["output"]).expanduser()
    if not out.is_absolute():
        out = root / out

    ensure_git_lfs()
    clone_url = build_clone_url(dataset_id)
    print(f"数据源:     {dataset_id}")
    print(f"Clone URL:  {clone_url}")
    print(f"输出目录:   {out}")
    print()

    git_clone(clone_url, out, branch, depth, skip_lfs)

    if preset_key == "yolo":
        print_yolo_usage(out)
    else:
        print_temporal_usage(out)


def print_temporal_usage(output: Path) -> None:
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
    print("      --config framework/experiments/gru-actionmixed.yaml")
    print(f"      （dataset_ref 已指向 {labels.parent}）")


def print_yolo_usage(output: Path) -> None:
    data_yamls = sorted(output.glob("*/data.yaml"))
    if not data_yamls:
        return
    print()
    print("--- YOLO 训练 ---")
    for data_yaml in data_yamls:
        print(f"  data.yaml: {data_yaml}")
    print("  Example (framework):")
    print("    python -m framework.cleansight_eval.cli.train \\")
    print("      --config framework/experiments/yolo-clean-large.yaml")
