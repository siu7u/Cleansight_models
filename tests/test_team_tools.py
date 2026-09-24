"""framework 组员友好能力（model_aliases / dataset_download / cli.dataset / cli.train --model）测试。

覆盖：
  - model_aliases：--list-models 内容、resolve_model 解析、未知模型报错
  - dataset_download：check_required_datasets 在临时目录下的缺失判定、
    已存在克隆的原地增量更新（不删目录）、非 git 非空目录的拒绝行为
  - cli.train：--model 与 --config 互斥、--list-models 无 torch 可用
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_list_models_contains_all():
    from framework.cleansight_eval.core.model_aliases import list_models

    text = list_models()
    for name in ("yolo", "gru", "mstcn", "mstcn2", "transformer", "feature_fusion",
                 "yolo11n", "yolo11s", "yolo11m"):
        assert name in text
    assert "--group" in text
    assert "cli.dataset --preset all" in text


def test_resolve_yolo_weights_and_group():
    from framework.cleansight_eval.core.model_aliases import resolve_model

    info = resolve_model("yolo11s", "group1_large")
    assert info["overrides"] == {"model.weights": "yolo11s.pt"}
    assert info["group"] == "group1_large"
    assert info["config"] == "yolo-clean-large.yaml"

    info2 = resolve_model("yolo", "group2_small")
    assert info2["group"] == "group2_small"
    assert "overrides" not in info2


def test_unknown_model_fails():
    from framework.cleansight_eval.core.model_aliases import resolve_model

    with pytest.raises(SystemExit, match="未知模型"):
        resolve_model("yolov5")


def test_model_config_path():
    from framework.cleansight_eval.core.model_aliases import model_config_path, resolve_model

    path = model_config_path(resolve_model("gru"))
    assert path.name == "gru-actionmixed.yaml"
    assert path.is_file()


def test_dataset_check_missing_in_tmp(tmp_path, monkeypatch):
    from framework.cleansight_eval.core.dataset_download import REQUIRED_FILES, check_required_datasets

    missing = check_required_datasets(["yolo", "actionmixed"], root=tmp_path)
    assert set(missing) == {"yolo", "actionmixed"}

    (tmp_path / "datasets/cleansight-yolo/group1_large").mkdir(parents=True)
    (tmp_path / "datasets/cleansight-yolo/group2_small").mkdir(parents=True)
    for rel in REQUIRED_FILES["yolo"]:
        (tmp_path / rel).write_text("train: images/train\n", encoding="utf-8")
    missing = check_required_datasets(["yolo", "actionmixed"], root=tmp_path)
    assert missing == ["actionmixed"]


def _run_git(args, cwd):
    """在 cwd 执行 git：显式身份并关闭签名，避免依赖（可能只读的）全局 config。"""

    return subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com",
         "-c", "commit.gpgsign=false", *args],
        cwd=cwd, capture_output=True, text=True, check=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def _make_upstream(tmp_path):
    """建本地 bare 上游 + 工作克隆（master 上已有一次提交），返回 (bare, work)。"""

    bare = tmp_path / "upstream.git"
    work = tmp_path / "upstream-work"
    subprocess.run(["git", "init", "--bare", str(bare)], capture_output=True, check=True)
    subprocess.run(["git", "init", str(work)], capture_output=True, check=True)
    _run_git(["symbolic-ref", "HEAD", "refs/heads/master"], work)
    (work / "labels").mkdir()
    (work / "labels/train.txt").write_text("v1\n", encoding="utf-8")
    _run_git(["add", "-A"], work)
    _run_git(["commit", "-m", "v1"], work)
    _run_git(["remote", "add", "origin", str(bare)], work)
    _run_git(["push", "-u", "origin", "master"], work)
    _run_git(["symbolic-ref", "HEAD", "refs/heads/master"], bare)
    return bare, work


def _commit_v2(work):
    """在上游再加一次提交，模拟数据集更新（改一个文件 + 新增一个文件）。"""

    (work / "labels/train.txt").write_text("v2\n", encoding="utf-8")
    (work / "labels/val.txt").write_text("new\n", encoding="utf-8")
    _run_git(["add", "-A"], work)
    _run_git(["commit", "-m", "v2"], work)
    _run_git(["push"], work)


def test_git_update_in_place_keeps_dir_and_untracked(tmp_path, capsys):
    """已存在克隆 → 原地增量更新：不删目录、覆盖脏文件、保留未跟踪文件。"""

    from framework.cleansight_eval.core.dataset_download import git_clone

    bare, work = _make_upstream(tmp_path)
    dest = tmp_path / "dest"
    git_clone(str(bare), dest, "master", 0, False)
    assert (dest / "labels/train.txt").read_text(encoding="utf-8") == "v1\n"

    # 模拟 LFS 空对象往返噪声（受版本控制文件被本地改动）+ 一个未跟踪文件
    (dest / "labels/train.txt").write_text("", encoding="utf-8")
    (dest / "local_note.txt").write_text("keep\n", encoding="utf-8")

    _commit_v2(work)
    git_clone(str(bare), dest, "master", 0, False)

    assert (dest / "labels/train.txt").read_text(encoding="utf-8") == "v2\n"
    assert (dest / "labels/val.txt").is_file()
    assert (dest / "local_note.txt").read_text(encoding="utf-8") == "keep\n"
    assert (dest / ".git").is_dir()
    assert _run_git(["rev-parse", "HEAD"], dest).stdout.strip() == (
        _run_git(["rev-parse", "master"], bare).stdout.strip()
    )

    # 上游无变化时再跑一次：幂等，报"已是最新"且不删目录
    capsys.readouterr()
    git_clone(str(bare), dest, "master", 0, False)
    assert "已是最新" in capsys.readouterr().out
    assert (dest / "labels/train.txt").read_text(encoding="utf-8") == "v2\n"


def test_git_update_shallow_clone_stays_shallow(tmp_path):
    """浅克隆同样能原地更新，且继续带 --depth，不退化成全量历史。"""

    from framework.cleansight_eval.core.dataset_download import git_clone

    bare, work = _make_upstream(tmp_path)
    dest = tmp_path / "dest"
    git_clone(f"file://{bare}", dest, "master", 1, False)
    _commit_v2(work)
    git_clone(f"file://{bare}", dest, "master", 1, False)

    assert (dest / "labels/train.txt").read_text(encoding="utf-8") == "v2\n"
    shallow = _run_git(["rev-parse", "--is-shallow-repository"], dest).stdout.strip()
    assert shallow == "true"


def test_git_clone_refuses_non_git_non_empty_dir(tmp_path):
    """非 git 的非空目录：明确报错并给出替代方案，绝不删除已有数据。"""

    from framework.cleansight_eval.core.dataset_download import git_clone

    bare, _ = _make_upstream(tmp_path)
    dest = tmp_path / "dest"
    (dest / "labels").mkdir(parents=True)
    (dest / "labels/train.txt").write_text("manual copy\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="不是 git 仓库"):
        git_clone(str(bare), dest, "master", 0, False)

    assert (dest / "labels/train.txt").read_text(encoding="utf-8") == "manual copy\n"


def test_train_cli_rejects_config_and_model_together():
    """--config 与 --model 同时传 → main 抛 SystemExit（互斥）。"""

    from framework.cleansight_eval.cli.train import main

    with pytest.raises(SystemExit, match="二选一"):
        main(["--config", "a.yaml", "--model", "gru"])


def test_train_cli_list_models_runs_without_torch():
    """--list-models 不 import torch/numpy 即可运行（在无 torch 的 python 下验证）。"""

    import subprocess
    import sys as _sys

    code = subprocess.run(
        [_sys.executable, "-m", "framework.cleansight_eval.cli.train", "--list-models"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert code.returncode == 0
    assert "可训练模型" in code.stdout
    assert "Traceback" not in code.stderr
