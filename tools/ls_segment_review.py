"""坏段复查工具（只读）：把"表现差的视频段"同时对齐到本地标签与 Label Studio 原标注。

用法::

    # ① LS 原标注 ↔ 本地标签 逐段对拍 + 标注质量扫描（重叠 / 零长 / <0.5s 碎段）
    python tools/ls_segment_review.py compare --split test

    # ② 从已存盘预测产物算逐视频 acc、逐类召回、逐 GT 段命中率（整段漏 / 误报 / 部分命中）
    python tools/ls_segment_review.py failures --arm flagship-v41

两条子命令都**只读**：不加载 checkpoint、不写回 Label Studio。产物写到
`outputs/ls_review/`（`outputs/` 不入库，可随时重生成）。

口径（与 v4.1 数据一致）：

- 本地 `labels/<split>/<video>.mp4.txt`：每行 `frame_id action_id`，`frame_id` 是 1-based
  **真实视频帧号**（30 fps 视频每 2 帧采 1 帧）→ 秒 = `frame_id / 30`。
- Label Studio `timelinelabels` 的 `ranges` 是 **24 fps 时间轴**的帧号 → 秒 = `frame / 24`；
  因此本地帧号 = LS 帧号 × 1.25（`compare` 会逐段打印该比值，用于发现重标定错误）。
- LS 访问凭证来自仓库根 `.env`（`LS_HOST` / `LS_TOKEN`）；`LS_TOKEN` 是 refresh token，
  先 POST `/api/token/refresh/` 换 access token（约 5 分钟有效）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import urllib.request
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO_ROOT, "datasets", "cleansight-ActionMixed-auto-lhh")
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "ls_review")
VIDEO_FPS = 30.0
LS_FPS = 24.0
ACTION_NAMES = ["idle", "water_injection", "flush", "long_brush_insert",
                "long_brush_withdraw", "short_brush_cleaning"]


# --------------------------------------------------------------------------- 公共

def _env() -> dict:
    env = {}
    with open(os.path.join(REPO_ROOT, ".env")) as fh:
        for line in fh:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


class LSClient:
    """只读 Label Studio 客户端（stdlib only）。"""

    def __init__(self) -> None:
        env = _env()
        self.host = env["LS_HOST"].rstrip("/")
        self._refresh = env["LS_TOKEN"]
        self._access = None

    def _token(self) -> str:
        if self._access is None:
            req = urllib.request.Request(
                self.host + "/api/token/refresh/",
                data=json.dumps({"refresh": self._refresh}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                self._access = json.load(resp)["access"]
        return self._access

    def get(self, path: str) -> dict:
        """GET 一个 LS API 路径（以 / 开头），返回解析后的 JSON。"""
        req = urllib.request.Request(self.host + path,
                                     headers={"Authorization": f"Bearer {self._token()}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)

    def task(self, task_id: int) -> dict:
        """取单个 task（含 annotations）。注意：列表接口不返回 annotations。"""
        return self.get(f"/api/tasks/{task_id}/")

    def link(self, project_id: int, task_id: int) -> str:
        """Label Studio 网页深链，会上可直接点开看该段。"""
        return f"{self.host}/projects/{project_id}/data?task={task_id}"


def load_task_ids(split: str) -> dict:
    """读 datasets/.../task_ids.yaml 里某个 split 的 {视频名: LS task id}。"""
    out, cur = {}, None
    with open(os.path.join(DATA_ROOT, "task_ids.yaml")) as fh:
        for line in fh:
            if line and not line[0].isspace() and line.rstrip().endswith(":"):
                cur = line.strip()[:-1]
            elif ":" in line and cur:
                k, v = line.strip().split(":", 1)
                v = v.split("#", 1)[0].strip()
                if v.isdigit() and cur == split:
                    out[k.strip()] = int(v)
    return out


def local_frames(split: str, video: str) -> tuple[list[int], list[int]]:
    """返回 (采样帧号, 动作 id) 两个平行列表。"""
    frames, labels = [], []
    with open(os.path.join(DATA_ROOT, "labels", split, f"{video}.txt")) as fh:
        for line in fh:
            line = line.strip()
            if line:
                fid, cid = line.split()
                frames.append(int(fid))
                labels.append(int(cid))
    return frames, labels


def to_runs(seq: list[int], frames: list[int]) -> list[dict]:
    """把逐帧类别序列压成段列表（含每段起止帧号与秒）。"""
    out, i = [], 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        out.append({
            "class_id": seq[i], "name": ACTION_NAMES[seq[i]],
            "start_frame": frames[i], "end_frame": frames[j],
            "num_frames": j - i + 1,
            "start_sec": round(frames[i] / VIDEO_FPS, 3),
            "end_sec": round(frames[j] / VIDEO_FPS, 3),
        })
        i = j + 1
    return out


def ls_ranges(annotation: dict) -> list[dict]:
    """从 LS annotation 取 timeline 段（ranges 单位 = LS 帧号）。"""
    out = []
    for r in annotation.get("result", []):
        if r.get("type") != "timelinelabels":
            continue
        labels = r.get("value", {}).get("timelinelabels") or []
        for rng in r.get("value", {}).get("ranges", []):
            out.append({"label": labels[0] if labels else None,
                        "start": rng["start"], "end": rng["end"]})
    out.sort(key=lambda x: x["start"])
    return out


# ------------------------------------------------------------------- 子命令 compare

def cmd_compare(args: argparse.Namespace) -> int:
    """LS 原标注 ↔ 本地标签 逐段对拍，并扫描标注质量问题。"""
    client = LSClient()
    task_ids = load_task_ids(args.split)
    rows, qa = [], []
    for video, tid in sorted(task_ids.items()):
        task = client.task(tid)
        anns = task.get("annotations") or []
        if not anns:
            print(f"[跳过] {video}: LS task {tid} 无标注")
            continue
        for ann in anns:
            ranges = ls_ranges(ann)
            pairs = [(ranges[i], ranges[j]) for i in range(len(ranges))
                     for j in range(i + 1, len(ranges)) if ranges[j]["start"] < ranges[i]["end"]]
            tiny = [r for r in ranges if r["end"] - r["start"] < 0.5 * LS_FPS]
            zero = [r for r in ranges if r["end"] <= r["start"]]
            qa.append({"split": args.split, "video": video, "task": tid,
                       "n_ranges": len(ranges), "n_overlap_pairs": len(pairs),
                       "n_zero_len": len(zero), "n_tiny": len(tiny),
                       "overlaps": "; ".join(f"{a['label']} {a['start']}-{a['end']} x "
                                             f"{b['label']} {b['start']}-{b['end']}" for a, b in pairs[:5]),
                       "link": client.link(task["project"], tid)})
            frames, labels = local_frames(args.split, video)
            local = [s for s in to_runs(labels, frames) if s["class_id"] != 0]
            ls_action = [r for r in ranges if r["label"] != "idle"]
            print(f"\n=== {args.split}/{video} (task {tid}, {len(ls_action)} LS 段 / {len(local)} 本地动作段)")
            by_label = defaultdict(list)
            for s in local:
                by_label[s["name"]].append(s)
            for r in ls_action:
                bucket = by_label.get(r["label"], [])
                # 按起点就近配对（避免顺序错位造成虚假比值）
                if bucket:
                    best = min(bucket, key=lambda s: abs(s["start_sec"] - r["start"] / LS_FPS))
                    ratio = best["start_sec"] / (r["start"] / LS_FPS) if r["start"] else float("nan")
                    bucket.remove(best)
                else:
                    best, ratio = None, float("nan")
                rows.append({
                    "split": args.split, "video": video, "task": tid, "label": r["label"],
                    "ls_start": r["start"], "ls_end": r["end"],
                    "ls_start_sec": round(r["start"] / LS_FPS, 3),
                    "local_start": best["start_frame"] if best else "",
                    "local_start_sec": best["start_sec"] if best else "",
                    "ratio_local_over_ls": round(ratio, 4) if best else "",
                    "link": client.link(task["project"], tid),
                })
                if best:
                    print(f"  {r['label']:22s} LS {r['start']:>5}-{r['end']:<5} "
                          f"({r['start']/LS_FPS:6.2f}-{r['end']/LS_FPS:6.2f}s) → 本地段 "
                          f"{best['start_sec']:7.3f}s 比值 {ratio:.4f}")
                else:
                    print(f"  {r['label']:22s} LS {r['start']:>5}-{r['end']:<5} "
                          f"({r['start']/LS_FPS:6.2f}-{r['end']/LS_FPS:6.2f}s) → **本地无对应段**")
            for q in qa:
                if q["video"] == video and (q["n_overlap_pairs"] or q["n_zero_len"]):
                    print(f"  !! 标注质量：重叠 {q['n_overlap_pairs']} 对 / 零长 {q['n_zero_len']} 段")

    os.makedirs(OUT_DIR, exist_ok=True)
    seg_csv = os.path.join(OUT_DIR, f"ls_vs_local_{args.split}.csv")
    with open(seg_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    qa_csv = os.path.join(OUT_DIR, f"ls_annotation_qa_{args.split}.csv")
    with open(qa_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qa[0].keys()))
        w.writeheader()
        w.writerows(qa)
    print(f"\n写出 {os.path.relpath(seg_csv, REPO_ROOT)}（{len(rows)} 段）"
          f"\n写出 {os.path.relpath(qa_csv, REPO_ROOT)}（{len(qa)} 条标注）")
    ratios = [r["ratio_local_over_ls"] for r in rows if r["ratio_local_over_ls"]]
    if ratios:
        print(f"比值中位 {statistics.median(ratios):.4f}（期望 ≈1.000 = 本地帧号/LS帧号 = 1.25 的时间轴换算）")
    return 0


# ------------------------------------------------------------------ 子命令 failures

def cmd_failures(args: argparse.Namespace) -> int:
    """从已存盘预测产物算逐视频 / 逐段失败画像（best.pt，取 acc 最接近中位的 seed）。"""
    base = os.path.join(REPO_ROOT, "runs", "v41", args.arm)
    if not os.path.isdir(base):
        print(f"没有这个臂目录：{base}", file=sys.stderr)
        return 2
    acc_by_stem, preds = {}, []
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".evaluation.json"):
                d = json.load(open(os.path.join(root, f)))
                if d["model"]["checkpoint"]["path"].endswith("best.pt"):
                    acc_by_stem[f[: -len(".evaluation.json")]] = d["metrics"]["summary"]["acc"]["value"]
            elif f.endswith(".predictions.json"):
                preds.append(os.path.join(root, f))
    pairs = [(acc_by_stem[os.path.basename(p)[: -len(".predictions.json")]], p) for p in preds
             if os.path.basename(p)[: -len(".predictions.json")] in acc_by_stem]
    if not pairs:
        print(f"{args.arm} 下找不到 (predictions.json, evaluation.json) 配对", file=sys.stderr)
        return 2
    pairs.sort()
    med = statistics.median([a for a, _ in pairs])
    acc, path = min(pairs, key=lambda x: abs(x[0] - med))
    print(f"[{args.arm}] {len(pairs)} 条 best.pt 评测，acc 中位 {med:.2f} → 取 acc={acc:.2f}\n"
          f"  {os.path.relpath(path, REPO_ROOT)}")

    pred = json.load(open(path))
    names = [l["name"] for l in pred["labels"]] or ACTION_NAMES
    rows, summary = [], {}
    for video, item in pred["items"].items():
        frames, gt = local_frames(args.split, video)
        pl = item["predicted_label_ids"]
        if len(gt) != len(pl):
            summary[video] = {"error": f"长度不一致 gt={len(gt)} pred={len(pl)}"}
            continue
        summary[video] = {"acc": round(sum(1 for a, b in zip(gt, pl) if a == b) / len(gt), 4)}
        for seg in to_runs(gt, frames):
            i0 = frames.index(seg["start_frame"])
            i1 = frames.index(seg["end_frame"])
            hit = sum(1 for i in range(i0, i1 + 1) if pl[i] == seg["class_id"])
            preds_in = sorted({names[pl[i]] for i in range(i0, i1 + 1)})
            rows.append({"video": video, "class": seg["name"], "start_sec": seg["start_sec"],
                         "end_sec": seg["end_sec"], "gt_frames": seg["num_frames"], "hit_frames": hit,
                         "coverage": round(hit / seg["num_frames"], 3),
                         "pred_in_seg": "|".join(preds_in),
                         "kind": "miss" if hit == 0 else ("partial" if hit / seg["num_frames"] < 0.5 else "ok")})
        for seg in to_runs(pl, frames):
            if seg["class_id"] == 0:
                continue
            i0 = frames.index(seg["start_frame"])
            i1 = frames.index(seg["end_frame"])
            gt_in = sorted({names[gt[i]] for i in range(i0, i1 + 1)})
            if names[seg["class_id"]] not in gt_in:
                rows.append({"video": video, "class": names[seg["class_id"]],
                             "start_sec": seg["start_sec"], "end_sec": seg["end_sec"],
                             "gt_frames": seg["num_frames"], "hit_frames": 0, "coverage": 0.0,
                             "pred_in_seg": "|".join(gt_in), "kind": "false_alarm"})

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"seg_failures_{args.arm}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n逐视频 acc：")
    for video, s in sorted(summary.items(), key=lambda kv: kv[1].get("acc", 1)):
        print(f"  {video[:12]:12s} {s.get('acc', s.get('error'))}")
    miss = [r for r in rows if r["kind"] == "miss" and r["class"] != "idle"]
    print(f"\n整段漏检（动作段，coverage=0）：{len(miss)} 段")
    for r in sorted(miss, key=lambda r: -r["gt_frames"])[:15]:
        print(f"  {r['video'][:10]:10s} {r['class']:22s} {r['start_sec']:7.2f}-{r['end_sec']:<7.2f}s "
              f"({r['gt_frames']:3d} 帧) 段内预测={r['pred_in_seg']}")
    fa = [r for r in rows if r["kind"] == "false_alarm"]
    print(f"误报段：{len(fa)} 个；部分命中段：{sum(1 for r in rows if r['kind'] == 'partial')} 个")
    print(f"\n写出 {os.path.relpath(out, REPO_ROOT)}（{len(rows)} 行）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="坏段复查：本地标签 × Label Studio 原标注 × 已存盘预测（只读）",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="LS 原标注 ↔ 本地标签 逐段对拍 + 标注质量扫描")
    c.add_argument("--split", default="test", help="数据集 split（须在 task_ids.yaml 里，默认 test）")
    c.set_defaults(func=cmd_compare)

    f = sub.add_parser("failures", help="逐视频/逐段失败画像（从已存盘预测产物算）")
    f.add_argument("--arm", default="flagship-v41", help="runs/v41/<arm>，默认旗舰臂")
    f.add_argument("--split", default="test", help="评测 split（默认 test）")
    f.set_defaults(func=cmd_failures)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
