"""数据集下载与就绪校验（framework 数据契约层）。

数据契约（catalog）只负责"登记与校验"；本模块负责把契约登记的数据集
从 ModelScope 下载到本地 `datasets/` 并在训练前校验就绪。CLI 入口见
``cli/dataset.py``（``python -m framework.cleansight_eval.cli.dataset``）。

下载逻辑从仓库根 `download_modelscope_dataset.py` 迁移而来（根脚本保留为薄封装，
向后兼容），但**目标目录已存在时的行为是原地增量更新**：走 `git fetch` +
`git reset --hard FETCH_HEAD`，任何失败都不删除数据目录（见 `_git_update`）。
"""

from __future__ import annotations

import os
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
    """把数据集放到 ``output``：已是 git 仓库则原地更新，否则克隆一份。

    参数：
        url: 克隆地址（ModelScope 数据集 git 地址，或测试用本地仓库路径）
        output: 目标目录；已存在 `.git` 时执行原地更新（不删目录）
        branch: 跟踪分支（ModelScope 数据集通常为 master）
        depth: 克隆/浅克隆增量抓取的深度，0 表示全量
        skip_lfs: 为 True 时设 GIT_LFS_SKIP_SMUDGE=1，只留 LFS 指针

    副作用：写文件系统；已有克隆时移动 HEAD（reset --hard），不删除任何数据目录。
    """

    if (output / ".git").is_dir():
        print(f"目标目录 {output} 已是 git 仓库，原地更新（fetch + reset --hard，不删目录）...")
        _git_update(output, branch, depth, skip_lfs)
        return

    if output.is_dir() and any(output.iterdir()):
        raise RuntimeError(
            f"目标目录 {output} 已存在且不是 git 仓库（无 .git），无法增量更新。\n"
            f"请勿直接删除数据，二选一：\n"
            f"  - 保留现有副本，改下载到新目录：--output <新目录>，"
            f"并同步 framework/testsets.yaml 的 data_root\n"
            f"  - 确认该副本可弃后，手动删除该目录再重跑本命令"
        )

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


def _rev(repo: Path) -> str:
    """返回仓库当前 HEAD 的 commit sha；取不到时返回空串。"""

    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _is_shallow(repo: Path) -> bool:
    """判断克隆是否为浅克隆（浅克隆必须继续带 --depth 抓增量，否则会拉回全历史）。"""

    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--is-shallow-repository"],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _git_update(output: Path, branch: str, depth: int, skip_lfs: bool) -> None:
    """把已有克隆原地更新到远端最新：``git fetch`` + ``git reset --hard FETCH_HEAD``。

    数据目录是只读资产，本地不存在需要保留的提交，所以用 reset --hard 而不是
    ``git merge --ff-only``：后者在远端改写历史、或工作区存在"本地修改"时会失败，
    而帧标签的 LFS 空对象（上游 pointer size 0）会让 git 恒报 modified。
    **本函数任何失败都只抛异常，不删除数据目录**，由使用者决定如何处理。
    """

    env = os.environ.copy()
    if skip_lfs:
        env["GIT_LFS_SKIP_SMUDGE"] = "1"

    old_rev = _rev(output)

    cmd = ["git", "-C", str(output), "fetch", "--no-tags"]
    if _is_shallow(output):
        cmd.extend(["--depth", str(depth)])
    cmd.extend(["origin", branch])
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"git fetch 失败 (exit {result.returncode})：\n{result.stderr}\n"
            f"数据目录未改动，也未删除：{output}"
        )

    result = subprocess.run(
        ["git", "-C", str(output), "reset", "--hard", "FETCH_HEAD"],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git reset --hard FETCH_HEAD 失败 (exit {result.returncode})：\n{result.stderr}\n"
            f"数据目录未被删除，可手动排查（git -C {output} status）：{output}"
        )

    new_rev = _rev(output)
    print(f"  更新完成: {output}")
    print(f"  revision: {old_rev[:8] or '(未知)'} → {new_rev[:8] or '(未知)'}")
    if old_rev and old_rev == new_rev:
        print("  已是最新，无数据变化。")
    else:
        _print_change_summary(output, old_rev, new_rev)
        _print_registration_hint()
    if not skip_lfs:
        _git_lfs_pull(output)
    _print_lfs_status(output)


def _print_change_summary(repo: Path, old_rev: str, new_rev: str) -> None:
    """打印本次更新引入的提交与变更统计（只读；浅克隆列不出提交时只给文件统计）。"""

    if not old_rev or not new_rev:
        return
    log = subprocess.run(
        ["git", "-C", str(repo), "log", "--oneline", f"{old_rev}..{new_rev}"],
        capture_output=True, text=True,
    )
    if log.returncode == 0:  # 浅克隆历史被截断时列不出提交，非致命
        lines = [line for line in log.stdout.strip().splitlines() if line.strip()]
        if lines:
            print(f"  新增提交 {len(lines)} 个（最近 {min(len(lines), 5)} 条）：")
            for line in lines[:5]:
                print(f"    {line}")
    stat = subprocess.run(
        ["git", "-C", str(repo), "diff", "--shortstat", old_rev, new_rev],
        capture_output=True, text=True,
    )
    if stat.returncode == 0 and stat.stdout.strip():
        print(f"  文件变更: {stat.stdout.strip()}")


def _print_registration_hint() -> None:
    """数据内容变化后的登记提醒（红线：数据内容变化必须走登记三件套）。"""

    print()
    print("  ⚠️  数据已变化，训练/评测前必须同步登记：")
    print("    1) 更新 framework/testsets.yaml 的 revision / dataset_version 与注释")
    print("    2) python tools/validate_testsets.py --catalog framework/testsets.yaml --json")
    print("    3) python -m framework.cleansight_eval.cli.dataset --check")
    print("    4) 源数据（frames/视频）变化时需重跑 embedding 等派生特征产物；")
    print("       旧 checkpoint 的 resume 会按内容指纹被拒（设计如此），新旧指标不可混比")


def _lfs_file_count(output: Path) -> int:
    """返回仓库中 LFS 跟踪文件数（git-lfs 不可用或无 LFS 时返回 0）。"""

    result = subprocess.run(
        ["git", "-C", str(output), "lfs", "ls-files", "--all"], capture_output=True, text=True,
    )
    if result.returncode != 0:
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _git_lfs_pull(output: Path) -> None:
    """补齐缺失的 LFS 内容；失败只告警，不让更新整体失败。"""

    if _lfs_file_count(output) == 0 or not check_git_lfs():
        return
    result = subprocess.run(
        ["git", "-C", str(output), "lfs", "pull"], capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  ⚠️  git lfs pull 未成功（LFS 大文件可能仍是指针）：{result.stderr.strip()[:200]}")


def _print_lfs_status(output: Path) -> None:
    count = _lfs_file_count(output)
    if count:
        print(f"  LFS 文件: {count} 个")


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
