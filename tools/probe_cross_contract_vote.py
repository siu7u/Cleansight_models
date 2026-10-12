"""跨契约 / 同契约**多成员投票**探针：读已存盘预测产物比对 acc、edit（不训练、不推理）。

动机：v4.1 全量重训显示**没有任何单一契约全能**——`bbox-global-hand` acc 最高（76.27）
但 flush/sbc 召回弱，`roi-grid-v5` edit 最高（69.43）且四类召回都 ≥0.55。逐帧标签的天然
融合方式是**多数投票**。本工具把两件事一次算清：

1. **同契约 8-seed 投票**：同一契约的多个 seed 投到一起（对照 = 该契约单 seed 中位）；
2. **跨契约两两投票**：两个契约各取一半成员投到一起（对照 = **两个契约中较强者的单 seed 中位**）。

后者是必须写死的口径：跨契约投票的成员池里天然混进了较弱契约的 seed，
若对照写成"成员平均"，会把"平均掉了弱成员"说成"集成有增益"（实测可虚高 +6~7pp）。

口径与纪律：
- 指标走 `framework.cleansight_eval.core.metrics.temporal_metrics`（与正式评测同一实现），
  `acc` = 逐帧 micro、`edit` = 逐视频 macro 平均；已核对与 `evaluation.json` 逐位一致；
- 投票 = 逐帧多数，平局取标签 id 较小者；`--merge-d` 可在投票后再做最小时长合并
  （复用 `tools/probe_offline_postprocess.py` 的 `merge_short`）；
- 配对检验在**逐视频**层面（n = 视频数），只作弱检验；**判 <1.5pp 的差异要靠 seed 数**，
  视频级 p 值不要当强证据读；
- **协议必须在 val 与 test 两侧同时看**：本仓库纪律是"test 涨、val 不支持一律不计交付"
  （见 `docs/EVAL.md` §9.2）。用 `--source val` 前需先把同一批 checkpoint 在 val 上复评
  （`benchmark.cli.eval` 默认只评 `data.split_eval`）。

用法：

    # test：11 个特征臂各 8 seed（run 目录直接给 glob）
    python tools/probe_cross_contract_vote.py \\
      --arm feat-roi-v5="runs/v41/feat-roi-v5/h128/mstcn2-*" \\
      --arm feat-global-hand="runs/v41/feat-global-hand/h128/mstcn2-*" \\
      --arm flagship-v41="runs/v41/flagship-v41/h128/mstcn2-*" --json tmp/vote.json

    # val：同一批 run 的 val 复评产物（含 artifacts/ 与 env.json 的目录皆可）
    python tools/probe_cross_contract_vote.py --source val \\
      --arm feat-roi-v5="tmp/v41push/val_runs/feat-roi-v5-*"

    # 投票 + 最小时长合并
    python tools/probe_cross_contract_vote.py --arm ... --merge-d 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="跨契约 / 同契约多成员投票探针（读产物，不训练）")
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=GLOB",
                        help="臂名=run 目录 glob（可多次）；glob 匹配到的每个 run 视为一个成员")
    parser.add_argument("--source", default="test", choices=("test", "val"),
                        help="产物来源标签（只用于打印与 JSON 记录；读取规则由 glob 决定）")
    parser.add_argument("--merge-d", type=int, default=0,
                        help="投票后再做最小时长合并（帧数；0=不做）。取样应先验给定")
    parser.add_argument("--pair-seeds", type=int, default=4,
                        help="跨契约投票时每臂取几个 seed（默认 4，两支共 8 成员）")
    parser.add_argument("--json", default=None)
    return parser.parse_args(argv)


def _read_run(run: Path) -> dict | None:
    """读一个 run 目录 → {seed, pred, truth, labels, acc, edit}；缺产物返回 None。

    预测产物必须与**该 run 最新一次正式评测**对齐：同一 run 的 `artifacts/` 下可能同时存在
    best.pt / last.pt 两次评测的产物（甚至不同 split 的复评产物），按文件名时间戳取最后一个
    会取错。这里优先用最新 `evals/*.evaluation.json` 的 `artifacts.predictions.path` 反查。
    """

    evals = sorted((run / "evals").glob("*.evaluation.json"))
    artifact = None
    summary = None
    if evals:
        document = json.loads(evals[-1].read_text())
        summary = document["metrics"]["summary"]
        reference = (document.get("artifacts") or {}).get("predictions") or {}
        if reference.get("path"):
            artifact = run / reference["path"]
    if artifact is None or not artifact.is_file():
        candidates = sorted((run / "artifacts").glob("*.predictions.json"))
        artifact = candidates[-1] if candidates else None
    if artifact is None:
        return None

    env = run / "env.json"
    seed_path = env if env.is_file() else (run / "seed.txt")
    if not seed_path.is_file():
        return None
    raw = seed_path.read_text()
    seed = json.loads(raw)["seed"] if seed_path.suffix == ".json" else int(raw.strip())
    doc = json.loads(artifact.read_text())
    labels = [entry["name"] for entry in doc["labels"]]
    index = {name: position for position, name in enumerate(labels)}
    pred = {v: np.array([index[x] for x in item["predicted_labels"]], dtype=np.int16)
            for v, item in doc["items"].items()}
    truth = {v: np.array([index[x] for x in item["truth_labels"]], dtype=np.int16)
             for v, item in doc["items"].items()}
    if summary:  # 指标真源 = 正式评测产物；缺失时本工具自算（口径一致）
        acc, edit = summary["acc"]["value"], summary["edit"]["value"]
    else:
        scored = score(pred, truth, labels)
        acc, edit = scored["acc"], scored["edit"]
    return {"seed": seed, "pred": pred, "truth": truth, "labels": labels, "acc": acc, "edit": edit}


def load_arms(specs: list[str]) -> tuple[dict, list[str]]:
    arms: dict[str, dict] = {}
    labels: list[str] = []
    for spec in specs:
        name, _, pattern = spec.partition("=")
        runs = sorted(REPO.glob(pattern))
        members = {}
        for run in runs:
            if not run.is_dir():
                continue
            record = _read_run(run)
            if record:
                members[record["seed"]] = record
        if members:
            arms[name] = members
            labels = labels or next(iter(members.values()))["labels"]
            print(f"[vote] {name:<20} {len(members)} 个成员 seeds={sorted(members)}")
        else:
            print(f"[vote] {name:<20} 无可用产物（跳过）：{pattern}")
    return arms, labels


def score(pred: dict, truth: dict, labels: list[str]) -> dict:
    """按官方口径重算 acc（逐帧 micro）与 edit（逐视频 macro 平均）。"""

    from framework.cleansight_eval.core.metrics import temporal_metrics

    metrics = temporal_metrics({v: [labels[int(x)] for x in pred[v]] for v in pred},
                               {v: [labels[int(x)] for x in truth[v]] for v in pred}, labels)
    return {"acc": metrics["frame"]["accuracy"] * 100, "edit": metrics["segment"]["edit"] * 100}


def vote(stack: np.ndarray, n_labels: int) -> np.ndarray:
    """``[N, T]`` 逐帧标签 → ``[T]`` 多数类（平局取标签 id 较小者）。"""

    tally = np.zeros((stack.shape[1], n_labels), dtype=np.int32)
    for row in stack:
        tally[np.arange(len(row)), row] += 1
    return tally.argmax(axis=1).astype(np.int16)


def per_video(pred: dict, truth: dict, labels: list[str]) -> dict:
    out = {}
    from framework.cleansight_eval.core.metrics import temporal_metrics

    for video in pred:
        item = temporal_metrics({video: [labels[int(x)] for x in pred[video]]},
                                {video: [labels[int(x)] for x in truth[video]]}, labels)
        out[video] = {"acc": float((pred[video] == truth[video]).mean() * 100),
                      "edit": item["segment"]["edit"] * 100}
    return out


def evaluate_members(members: list[tuple[str, int]], arms: dict, videos: list[str],
                     labels: list[str], merge_d: int = 0) -> dict:
    """投票评测，并给出**逐视频**配对（投票 vs 成员平均）。"""

    merge = None
    if merge_d and merge_d > 1:
        from probe_offline_postprocess import merge_short

        merge = lambda seq: merge_short(seq, merge_d)  # noqa: E731
    voted = {}
    for video in videos:
        stacked = np.stack([arms[a][s]["pred"][video] for a, s in members])
        result = vote(stacked, len(labels))
        voted[video] = merge(result) if merge else result
    truth = {v: arms[members[0][0]][members[0][1]]["truth"][v] for v in videos}
    scored = score(voted, truth, labels)
    member_video = {}
    for arm, seed in members:
        record = arms[arm][seed]
        for video, value in per_video(record["pred"], record["truth"], labels).items():
            member_video.setdefault(video, []).append(value)
    ensemble_video = per_video(voted, truth, labels)
    out = {"acc": scored["acc"], "edit": scored["edit"], "n_members": len(members)}
    for key in ("acc", "edit"):
        diffs = [ensemble_video[v][key] - statistics.mean(x[key] for x in member_video[v])
                 for v in videos]
        out[f"d_{key}"] = statistics.median(diffs)
        out[f"w_{key}"] = sum(1 for d in diffs if d > 0)
        try:
            out[f"p_{key}"] = float(wilcoxon(diffs).pvalue)
        except ValueError:
            out[f"p_{key}"] = float("nan")
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    specs = args.arm if isinstance(args.arm, list) else [args.arm]
    arms, labels = load_arms(specs)
    if not arms:
        print("[vote] 未匹配到任何含 predictions 的 run")
        return 2
    videos = sorted(next(iter(next(iter(arms.values())).values()))["pred"])
    print(f"[vote] source={args.source} 臂={len(arms)} 视频={len(videos)} merge_d={args.merge_d}")

    print("\n## 1. 单 seed 中位（各臂自身对照）")
    print(f"{'arm':<20}{'n':>4}{'acc':>8}{'edit':>8}")
    single = {}
    for name, members in arms.items():
        acc = statistics.median(m["acc"] for m in members.values())
        edit = statistics.median(m["edit"] for m in members.values())
        single[name] = {"acc": acc, "edit": edit, "n": len(members)}
        print(f"{name:<20}{len(members):>4}{acc:>8.2f}{edit:>8.2f}")

    print("\n## 2. 同契约全成员投票 vs 该臂单 seed 中位")
    print(f"{'arm':<20}{'acc':>8}{'Δacc':>8}{'edit':>8}{'Δedit':>8}{'胜acc':>7}{'胜edit':>7}")
    same = {}
    for name, members in arms.items():
        result = evaluate_members([(name, s) for s in sorted(members)], arms, videos, labels,
                                  args.merge_d)
        same[name] = result
        print(f"{name:<20}{result['acc']:>8.2f}{result['acc'] - single[name]['acc']:>+8.2f}"
              f"{result['edit']:>8.2f}{result['edit'] - single[name]['edit']:>+8.2f}"
              f"{result['w_acc']:>7}{result['w_edit']:>7}")

    print(f"\n## 3. 跨契约两两投票（每臂 {args.pair_seeds} seed）vs 较强父契约单 seed 中位")
    print(f"{'A + B':<40}{'acc':>8}{'Δacc':>8}{'edit':>8}{'Δedit':>8}{'胜acc':>7}{'胜edit':>7}")
    pairs = []
    for a, b in itertools.combinations(sorted(arms), 2):
        members = ([(a, s) for s in sorted(arms[a])[:args.pair_seeds]]
                   + [(b, s) for s in sorted(arms[b])[:args.pair_seeds]])
        result = evaluate_members(members, arms, videos, labels, args.merge_d)
        base_acc = max(single[a]["acc"], single[b]["acc"])
        base_edit = max(single[a]["edit"], single[b]["edit"])
        pairs.append({"a": a, "b": b, **result, "base_acc": base_acc, "base_edit": base_edit})
    pairs.sort(key=lambda item: -(item["acc"] - item["base_acc"] + item["edit"] - item["base_edit"]))
    for item in pairs:
        print(f"{item['a'] + ' + ' + item['b']:<40}{item['acc']:>8.2f}"
              f"{item['acc'] - item['base_acc']:>+8.2f}{item['edit']:>8.2f}"
              f"{item['edit'] - item['base_edit']:>+8.2f}{item['w_acc']:>7}{item['w_edit']:>7}")

    if pairs:
        print("\n  Δ 汇总（相对较强父契约单 seed 中位）：")
        for key in ("acc", "edit"):
            diffs = [item[key] - item[f"base_{key}"] for item in pairs]
            print(f"    {key}: 中位 {statistics.median(diffs):+.2f} 均值 {statistics.mean(diffs):+.2f} "
                  f"正向 {sum(1 for d in diffs if d > 0)}/{len(diffs)} p={wilcoxon(diffs).pvalue:.3g}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"source": args.source, "merge_d": args.merge_d,
                                    "single": single, "same_contract": same, "pairs": pairs},
                                   ensure_ascii=False, indent=1))
        print(f"\n[vote] 结果已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
