#!/usr/bin/env python3
"""构建「自动标注 + 手动标注」时序合并训练集（`cleansight-ActionMixed-union`）。

**动机**（2026-09-29，帧准确率攻关第 2 轮）：现行旗舰在 8 视频 test 上的帧准确率
`acc 53.92` 低于平凡基线（全 idle `55.25`）。误差分解显示模型**先把动作帧判成 idle**
（786/1181），而两个失效模式都指向训练数据：
① 跨批次——train/val 来自 Label Studio `project-16`（2026-08），test 来自 `project-18`
   （2026-09，长短毛刷专项）；
② 先验偏 idle——auto train 的 idle 占 65.5%，而 test 只有 55.2%。
手动数据集 `datasets/cleansight-ActionMixed` 提供 **16 个与 auto 零交集的视频 / 9,532 帧**，
且 **idle 只占约 25.6%**，是同时针对这两个失效模式的额外监督。

**这个脚本产出什么**：一个目录形态与 auto 数据集完全一致（`frames/<split>/*.txt` +
`labels/<split>/<video>.mp4.txt` + 两份 `data.yaml`）的合并根：

| split | 内容 | 实现 |
|---|---|---|
| `train` | auto-train 14 视频 + 手动集 **全部 16 视频的全部帧** | 帧文件**逐文件符号链接**；标签按 stem 合并（手动集的三个 split 是同一视频的不同帧区间，合并时取并集） |
| `val` / `test` | 与 auto 的 val/test **逐字节相同** | 整目录符号链接 → 选点与验收口径与基线完全一致，保证单变量对照 |

**两条不变量**（脚本内置断言，违反直接报错）：

1. 两个来源的视频 stem **零交集**（否则同一 stem 的帧文件会互相覆盖）；
2. 手动集内部 train/val/test 对同一 stem 的帧号**零交集**（否则合并出的标签会重复）。

**不会修改任何原始数据集目录**：`cleansight-ActionMixed` 与 `cleansight-ActionMixed-auto-lhh`
都只读。合并集是**派生产物**，可随时删掉重跑。

用法：

    python tools/build_union_temporal_dataset.py            # 构建（幂等，已存在则先删）
    python tools/build_union_temporal_dataset.py --check    # 只做一致性检查与统计，不写盘
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

AUTO_ROOT = REPO / "datasets/cleansight-ActionMixed-auto-lhh"
MANUAL_ROOT = REPO / "datasets/cleansight-ActionMixed"
UNION_ROOT = REPO / "datasets/cleansight-ActionMixed-union"

SPLITS = ("train", "val", "test")


def read_label_file(path: Path) -> list[tuple[int, int]]:
    """读 `"frame_id action_id"` 行 → 有序 ``[(frame_id, action_id)]``（跳过畸形行）。"""

    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        rows.append((int(parts[0]), int(parts[1])))
    return rows


def stems_of(root: Path, split: str) -> list[str]:
    """某 split 下 `labels/` 里的视频 stem（去掉 ``.txt``，保留 ``<video>.mp4``）。"""

    return sorted(path.name[:-4] for path in (root / "labels" / split).glob("*.txt"))


def collect_manual(manual_root: Path) -> dict[str, list[tuple[int, int, str]]]:
    """手动集 → ``{stem: [(frame_id, action_id, 来源 split), ...]}``，按帧号排序去重。

    手动集的 train/val/test 是**同一批视频的不同帧区间**，因此按 stem 汇总时要跨 split 取并集；
    帧号重复（跨 split 重叠）会破坏"帧互斥"不变量，直接报错。
    """

    merged: dict[str, list[tuple[int, int, str]]] = {}
    seen: dict[str, dict[int, str]] = {}
    for split in SPLITS:
        for stem in stems_of(manual_root, split):
            for frame_id, action_id in read_label_file(manual_root / "labels" / split / f"{stem}.txt"):
                owner = seen.setdefault(stem, {})
                if frame_id in owner:
                    raise SystemExit(
                        f"[FAIL] 手动集内帧号跨 split 重叠: stem={stem} frame={frame_id} "
                        f"({owner[frame_id]} 与 {split})——合并会产生重复标签"
                    )
                owner[frame_id] = split
                merged.setdefault(stem, []).append((frame_id, action_id, split))
    for stem in merged:
        merged[stem].sort()
    return merged


def build(check_only: bool) -> int:
    if not AUTO_ROOT.is_dir() or not MANUAL_ROOT.is_dir():
        raise SystemExit(f"[FAIL] 数据源缺失: {AUTO_ROOT} / {MANUAL_ROOT}")

    auto_train = stems_of(AUTO_ROOT, "train")
    manual = collect_manual(MANUAL_ROOT)
    manual_stems = sorted(manual)

    overlap = set(auto_train) & set(manual_stems)
    if overlap:
        raise SystemExit(f"[FAIL] 两个来源的视频 stem 有交集（帧文件会互相覆盖）: {sorted(overlap)[:3]}")

    # ---- 统计 ----
    auto_frames = sum(len(read_label_file(AUTO_ROOT / "labels/train" / f"{s}.txt")) for s in auto_train)
    manual_frames = sum(len(rows) for rows in manual.values())
    auto_actions: dict[int, int] = {}
    for stem in auto_train:
        for _fid, action in read_label_file(AUTO_ROOT / "labels/train" / f"{stem}.txt"):
            auto_actions[action] = auto_actions.get(action, 0) + 1
    manual_actions: dict[int, int] = {}
    for rows in manual.values():
        for _fid, action, _split in rows:
            manual_actions[action] = manual_actions.get(action, 0) + 1

    def dist(counts: dict[int, int]) -> str:
        total = sum(counts.values()) or 1
        return " ".join(f"id{k}={v}({100 * v / total:.1f}%)" for k, v in sorted(counts.items()))

    print(f"[统计] auto train  : {len(auto_train)} 视频 / {auto_frames} 帧  {dist(auto_actions)}")
    print(f"[统计] manual 全集 : {len(manual_stems)} 视频 / {manual_frames} 帧  {dist(manual_actions)}")
    union_count: dict[int, int] = dict(auto_actions)
    for k, v in manual_actions.items():
        union_count[k] = union_count.get(k, 0) + v
    print(f"[统计] 合并 train  : {len(auto_train) + len(manual_stems)} 视频 / "
          f"{auto_frames + manual_frames} 帧  {dist(union_count)}")

    if check_only:
        print("[check] 一致性检查通过（未写盘）")
        return 0

    # ---- 构建 ----
    if UNION_ROOT.exists():
        shutil.rmtree(UNION_ROOT)
    for split in SPLITS:
        (UNION_ROOT / "frames" / split).mkdir(parents=True)
        (UNION_ROOT / "labels" / split).mkdir(parents=True)

    # train：auto 逐文件链接 + 标签复制；manual 逐文件链接 + 标签合并
    n_links = 0
    for stem in auto_train:
        rows = read_label_file(AUTO_ROOT / "labels/train" / f"{stem}.txt")
        (UNION_ROOT / "labels/train" / f"{stem}.txt").write_text(
            "\n".join(f"{fid} {act}" for fid, act in rows) + "\n")
        for fid, _act in rows:
            src = AUTO_ROOT / "frames/train" / f"{stem}-{fid:06d}.txt"
            if not src.is_file():
                raise SystemExit(f"[FAIL] 缺少帧文件: {src}")
            (UNION_ROOT / "frames/train" / src.name).symlink_to(src)
            n_links += 1
    for stem, rows in manual.items():
        (UNION_ROOT / "labels/train" / f"{stem}.txt").write_text(
            "\n".join(f"{fid} {act}" for fid, act, _split in rows) + "\n")
        for fid, _act, split in rows:
            src = MANUAL_ROOT / "frames" / split / f"{stem}-{fid:06d}.txt"
            if not src.is_file():
                raise SystemExit(f"[FAIL] 缺少帧文件: {src}")
            (UNION_ROOT / "frames/train" / src.name).symlink_to(src)
            n_links += 1

    # val / test：整目录符号链接 → 与 auto 逐字节相同
    for split in ("val", "test"):
        (UNION_ROOT / "frames" / split).rmdir()
        (UNION_ROOT / "labels" / split).rmdir()
        (UNION_ROOT / "frames" / split).symlink_to(AUTO_ROOT / "frames" / split)
        (UNION_ROOT / "labels" / split).symlink_to(AUTO_ROOT / "labels" / split)

    # 类别表：动作表用 auto 版（id 1 = water_injection，与 benchmark 的类名一致；
    # 手动集只是同一动作的 LS 更名 air_injection，id 位置不变）
    shutil.copyfile(AUTO_ROOT / "labels/data.yaml", UNION_ROOT / "labels/data.yaml")
    shutil.copyfile(AUTO_ROOT / "frames/data.yaml", UNION_ROOT / "frames/data.yaml")

    # 合并集 revision：与 auto 条目同口径——三个 split manifest 拼接内容的 sha256（由
    # benchmark/manifests 决定），此处只记 train 标签内容的 sha256 作为构建指纹。
    digest = hashlib.sha256()
    for stem in sorted(auto_train) + manual_stems:
        digest.update((UNION_ROOT / "labels/train" / f"{stem}.txt").read_bytes())
    print(f"[构建] {UNION_ROOT}")
    print(f"[构建] train 视频 {len(auto_train) + len(manual_stems)}（auto {len(auto_train)} + "
          f"manual {len(manual_stems)}）/ 符号链接 {n_links} 个帧文件")
    print(f"[构建] train 标签内容 sha256 = {digest.hexdigest()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 auto+manual 时序合并训练集")
    parser.add_argument("--check", action="store_true", help="只做一致性检查与统计，不写盘")
    args = parser.parse_args()
    return build(args.check)


if __name__ == "__main__":
    sys.exit(main())
