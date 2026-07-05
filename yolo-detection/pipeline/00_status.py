#!/usr/bin/env python3
"""
对账 / 增量前置:把四方来源对齐,告诉你每次该做什么。

四方:
  - LS 导出 JSON(raw/exports/ 最新一份)—— 标注侧"应该有"的视频
  - raw/videos/ ——   实际下载到磁盘的视频
  - splits.yaml ——    已归属 split 的视频
  - config.only_videos —— 已人工质检合格(白名单)

输出一张状态表 + 可执行分类:
  未下载  导出引用了但磁盘没有        -> 跑 01_pull_data.py
  孤儿    磁盘有但导出没引用          -> 陈旧下载,可清理
  未质检  已下载但不在白名单          -> 人工质检后追加 config.only_videos
  未归属  已质检但 splits.yaml 没分配 -> 跑本脚本 --assign 回填
  遗失    splits.yaml 有但磁盘没有    -> 悬挂项,重下或从 splits.yaml 删(不自动删)

用法(在 yolo_pipeline/ 下执行):
  python3 00_status.py            # 只读,打印状态
  python3 00_status.py --assign   # 给"未归属"确定性回填 split 并写回 splits.yaml
"""
import sys

from utils.common import load_config, is_whitelisted
from utils import lsexport, split as splitmod


def gather():
    cfg = load_config()
    only = cfg.get("only_videos") or []
    label2group = lsexport.build_label_index(cfg["groups"])

    json_path = lsexport.latest_export()
    tasks = lsexport.load_tasks(json_path)
    sp = splitmod.load()
    assignments = sp.get("assignments", {})

    # stem -> 记录
    rows = {}

    def row(stem):
        return rows.setdefault(stem, {
            "name": stem, "in_export": False, "on_disk": False,
            "whitelisted": False, "has_det": False, "split": None,
        })

    # 导出侧
    for t in tasks:
        name = lsexport.task_video_name(t)
        if not name:
            continue
        r = row(splitmod.stem_of(name))
        r["name"] = name
        r["in_export"] = True
        r["whitelisted"] = is_whitelisted(name, only)
        if lsexport.collect_tracks(t, label2group):
            r["has_det"] = True

    # 磁盘侧
    for f in sorted(lsexport.VIDEO_DIR.glob("*.mp4")):
        r = row(splitmod.stem_of(f.name))
        r["name"] = f.name
        r["on_disk"] = True
        if not r["in_export"]:  # 孤儿也判一下白名单以免漏显
            r["whitelisted"] = is_whitelisted(f.name, only)

    # splits 侧
    for stem, s in assignments.items():
        row(stem)["split"] = s

    return cfg, json_path, sp, rows


def classify(rows):
    cats = {"未下载": [], "孤儿": [], "未质检": [], "未归属": [], "遗失": []}
    for stem, r in rows.items():
        if r["in_export"] and not r["on_disk"]:
            cats["未下载"].append(r)
        if r["on_disk"] and not r["in_export"]:
            cats["孤儿"].append(r)
        if r["on_disk"] and not r["whitelisted"]:
            cats["未质检"].append(r)
        if r["on_disk"] and r["whitelisted"] and r["split"] is None:
            cats["未归属"].append(r)
        if r["split"] is not None and not r["on_disk"]:
            cats["遗失"].append(r)
    return cats


def print_table(rows):
    def yn(b):
        return "✓" if b else "·"
    print(f"\n{'视频 (stem)':<52} {'导出':<4} {'磁盘':<4} {'质检':<4} {'检测':<4} {'split'}")
    print("-" * 82)
    for stem in sorted(rows):
        r = rows[stem]
        print(f"{stem:<52} {yn(r['in_export']):<4} {yn(r['on_disk']):<4} "
              f"{yn(r['whitelisted']):<4} {yn(r['has_det']):<4} {r['split'] or '—'}")


def print_cats(cats):
    hints = {
        "未下载": "跑 01_pull_data.py",
        "孤儿": "陈旧下载,可清理",
        "未质检": "人工质检后追加 config.yaml 的 only_videos",
        "未归属": "跑 00_status.py --assign 回填",
        "遗失": "重下,或从 splits.yaml 删除(不自动删)",
    }
    print("\n=== 待办分类 ===")
    any_action = False
    for cat in ["未下载", "未质检", "未归属", "遗失", "孤儿"]:
        items = cats[cat]
        if not items:
            continue
        any_action = True
        print(f"\n[{cat}] {len(items)} 个 —— {hints[cat]}")
        for r in items:
            print(f"    {r['name']}")
    if not any_action:
        print("  一切就绪,无待办。")


def main():
    do_assign = "--assign" in sys.argv[1:]
    cfg, json_path, sp, rows = gather()
    print(f"导出: {json_path.name}   视频目录: {lsexport.VIDEO_DIR}")
    print_table(rows)
    cats = classify(rows)
    print_cats(cats)

    if do_assign:
        pending = [r["name"] for r in cats["未归属"]]
        stems = [splitmod.stem_of(n) for n in pending]
        added = splitmod.assign(stems, sp)
        if added:
            splitmod.save(sp)
            print(f"\n=== --assign 回填 {len(added)} 个(已写回 splits.yaml)===")
            for stem, s in added:
                print(f"    {stem} -> {s}")
            print("请 review 并提交 splits.yaml 的改动。")
        else:
            print("\n=== --assign:没有需要回填的视频 ===")


if __name__ == "__main__":
    main()
