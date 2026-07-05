#!/usr/bin/env python3
"""
LS 导出 JSON + 视频 -> YOLO 目标检测数据集,按 splits.yaml 稳定切分(整段路由)。

与旧 01_to_yolo.py 的唯一行为差别:train/val 不再按抽帧序号切,而是**按视频**——
一个视频的所有帧全部进它在 splits.yaml 里的 split,杜绝时间相邻泄漏、且可复现。

前置:所有"已质检"的视频都要在 splits.yaml 里有归属。遇到未归属的视频默认报错,
提示先跑 `00_status.py --assign`;传 --auto-assign 可当场回填并写回 splits.yaml。

用法(在 yolo_pipeline/ 下执行):
    python3 02_build_dataset.py
    python3 02_build_dataset.py --auto-assign
"""
import shutil
import sys
from collections import defaultdict

import cv2

from utils.common import ROOT, load_config, is_whitelisted
from utils import lsexport, split as splitmod, stats

OUT_ROOT = ROOT / "datasets"


def prepare_dirs(groups):
    # 每次全量重建:先清空各组 images/labels,避免旧产物(可能来自不同切分)残留导致跨 split 泄漏。
    for g in groups:
        for sub in ("images", "labels"):
            shutil.rmtree(OUT_ROOT / g / sub, ignore_errors=True)
        for s in ("train", "val"):
            (OUT_ROOT / g / "images" / s).mkdir(parents=True, exist_ok=True)
            (OUT_ROOT / g / "labels" / s).mkdir(parents=True, exist_ok=True)


def write_data_yaml(groups):
    for g, labels in groups.items():
        names = "\n".join(f"  {i}: {lab}" for i, lab in enumerate(labels))
        (OUT_ROOT / g / "data.yaml").write_text(
            f"path: {(OUT_ROOT / g).resolve()}\n"
            f"train: images/train\nval: images/val\n"
            f"nc: {len(labels)}\nnames:\n{names}\n",
            encoding="utf-8",
        )


def main():
    auto_assign = "--auto-assign" in sys.argv[1:]
    cfg = load_config()
    groups = cfg["groups"]
    only = cfg.get("only_videos") or []
    label2group = lsexport.build_label_index(groups)
    stride = cfg.get("stride", 12)
    jpg_q = cfg.get("jpg_quality", 90)

    json_path = lsexport.latest_export()
    tasks = lsexport.load_tasks(json_path)
    sp = splitmod.load()
    print(f"导出: {json_path.name}  共 {len(tasks)} 个 task")

    # 先确定每个"待处理视频"的 split;缺归属就拦下(除非 --auto-assign)
    pending = []  # (task_index, task, name, split)
    unassigned = []
    for ti, task in enumerate(tasks):
        name = lsexport.task_video_name(task)
        if not name or not is_whitelisted(name, only):
            continue
        if not lsexport.collect_tracks(task, label2group):
            continue  # 无分组内目标
        stem = splitmod.stem_of(name)
        s = splitmod.get_split(stem, sp)
        if s is None:
            unassigned.append((stem, ti, task, name))
        else:
            pending.append((ti, task, name, s))

    if unassigned:
        stems = [u[0] for u in unassigned]
        if auto_assign:
            added = splitmod.assign(stems, sp)
            splitmod.save(sp)
            print(f"[auto-assign] 回填 {len(added)} 个到 splits.yaml: "
                  + ", ".join(f"{k}->{v}" for k, v in added))
            for stem, ti, task, name in unassigned:
                pending.append((ti, task, name, splitmod.get_split(stem, sp)))
        else:
            print("✗ 以下已质检视频在 splits.yaml 无归属,先跑 "
                  "`python3 00_status.py --assign`(或本脚本加 --auto-assign):")
            for stem, *_ in unassigned:
                print(f"    {stem}")
            sys.exit(2)

    prepare_dirs(groups)
    emitted = 0

    for ti, task, name, split in pending:
        if split not in splitmod.TRAINVAL_SPLITS:
            print(f"  [hold] task#{ti} {name} split={split},保留不进数据集")
            continue
        vpath = lsexport.VIDEO_DIR / name
        if not vpath.exists():
            print(f"  [warn] task#{ti} 视频缺失,跳过: {name}")
            continue

        tracks = lsexport.collect_tracks(task, label2group)
        cap = cv2.VideoCapture(str(vpath))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        real_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fc, dur = lsexport.clip_meta(task)
        scale, ls_fps = lsexport.fps_scale(real_fps, fc, dur)
        stem12 = vpath.stem[:12]
        print(f"  task#{ti} [{split}] {name}  真实帧={total}@{real_fps:.2f}  "
              f"LS={fc}@{ls_fps:.2f}  scale={scale:.4f}  轨迹={len(tracks)}")

        frame_idx = 0
        max_sampled_real = 0
        while True:
            if not cap.grab():
                break
            frame_idx += 1              # LS frame 从 1 开始
            if (frame_idx - 1) % stride != 0:
                continue
            ls_frame = frame_idx * scale
            lines_by_group = defaultdict(list)
            for g, cid, segs in tracks:
                box = lsexport.box_at(segs, ls_frame)
                if box is None:
                    continue
                cx, cy, w, h = lsexport.to_yolo(*box)
                if w <= 0 or h <= 0:
                    continue
                lines_by_group[g].append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if not lines_by_group:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            max_sampled_real = frame_idx
            base = f"{ti:02d}_{stem12}_{frame_idx:06d}"
            for g, lines in lines_by_group.items():
                cv2.imwrite(str(OUT_ROOT / g / "images" / split / f"{base}.jpg"),
                            frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_q])
                (OUT_ROOT / g / "labels" / split / f"{base}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8")
                emitted += 1
        cap.release()
        # 尾部覆盖自查:抽帧最大真实帧号应接近总帧数
        cover = (max_sampled_real / total * 100) if total else 0
        flag = "  ⚠️尾部可能丢失" if total and max_sampled_real < total * 0.8 else ""
        print(f"        尾部覆盖: 最大抽帧真实帧号={max_sampled_real}/{total} "
              f"({cover:.0f}%){flag}")

    write_data_yaml(groups)

    # 样本分布(训练帧粒度):扫描刚落盘的 label,反映磁盘真实结果
    for g, class_names in groups.items():
        stats.print_distribution(g, class_names, OUT_ROOT / g)
    print(f"\n共写出图像 {emitted} 张。data.yaml 已生成。下一步:03_train.py / 04_validate.py")


if __name__ == "__main__":
    main()
