"""ROI 契约的**通道语义 / 空间区域子集消融**探针（纯分析，不训练）。

问题：`roi-grid-144` 的每一类 18 维 = 6 区域 × `[presence, count, max_area]`。到底哪一部分
在携带判别力？如果 `count`/`max_area` 冗余，契约可以缩到 48 维（presence-only）；
如果某些网格区域从不贡献，也可以砍掉。这属于"特征提取方式"的**配方内部**维度，
与"换契约"（`temporal.actionmixed-auto-roi-v2` 等）是两条独立的路。

**注意与 `feature_schema.mask_targets` 的区别**：后者只能遮**整类块**（18 维）；
本探针做的是**通道/区域级**子集，用纯分析（不新增配置面）回答同一类问题。
若要把它变成可训练契约，需要新增 feature mapping——先用本探针判定值不值得。

口径（与正式评测一致，便于对照）：
- 逐帧多类 LDA（train 拟合，闭式解，不训练），test 上**离线协议**：逐帧 argmax、零平滑、无冷启动；
- 指标走 `framework.cleansight_eval.core.metrics.temporal_metrics`；子集之间比较用
  **逐视频配对 Wilcoxon**（同一 LDA 家族、同一批视频，配对合法）；
- 通道索引按契约声明：类块 18 维、区域行优先、每区域 3 通道 → ``k*18 + r*3 + c``。

用法：
    python tools/probe_channel_subsets.py --config-run "runs/mstcn2-cap-s2l5h128/*/mstcn2-*"
    python tools/probe_channel_subsets.py --config-run "<run glob>" --json tmp/channel_subsets.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CHANNEL_NAMES = ("presence", "count", "max_area")


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="ROI 契约的通道语义 / 区域子集消融（LDA 探针，不训练）")
    parser.add_argument("--config-run", required=True, help="用于解析 data/feature_schema 的 run glob（相对仓库根）")
    parser.add_argument("--json", default=None, help="结果写到此 JSON")
    return parser.parse_args(argv)


def channel_indices(n_classes: int, rows: int, cols: int, channels: int, *,
                    keep_channels: Sequence[int] | None = None,
                    keep_regions: Sequence[int] | None = None) -> np.ndarray:
    """按 `k*block + r*channels + c`（block = rows*cols*channels，区域行优先）取子集通道索引。"""
    block = rows * cols * channels
    channel_set = set(range(channels)) if keep_channels is None else set(keep_channels)
    region_set = set(range(rows * cols)) if keep_regions is None else set(keep_regions)
    indices = [k * block + r * channels + c
               for k in range(n_classes)
               for r in range(rows * cols) if r in region_set
               for c in range(channels) if c in channel_set]
    return np.array(sorted(indices), dtype=int)


def read_contract(config_glob: str) -> tuple[dict, dict, dict]:
    """读配置的 feature_schema/model 段，并从 **catalog** 取该 dataset_ref 的 layout。

    layout（rows/cols/channels 或 groups）在 resolved config 里**不存在**，只在
    `framework/testsets.yaml` 登记；早先版本用代码默认值（2×3×3）凑巧对 roi-144 成立，
    对其它网格契约（如 presence-only 48 维）会算错块宽——这里改为从 catalog 读，单一事实源。
    """
    from framework.cleansight_eval.core.catalog import get_dataset_specs
    from framework.cleansight_eval.core.config import load_config

    run_dir = sorted(REPO.glob(config_glob))[0]
    cfg = load_config(run_dir / "config.resolved.json")
    layout = dict(cfg["feature_schema"].get("layout") or {})
    dataset_ref = (cfg.get("data") or {}).get("dataset_ref")
    if not layout and dataset_ref:
        for spec in get_dataset_specs(str(dataset_ref)):
            catalog_layout = spec.raw.get("feature_layout")
            if catalog_layout:
                layout = dict(catalog_layout)
                break
    return cfg["feature_schema"], cfg["model"], layout


def load_sequences(config_glob: str):
    """→ (train_feats, train_truths, test_feats, test_truths, names, labels, feature_schema)"""
    from framework.cleansight_eval.core.config import load_config
    from framework.cleansight_eval.temporal.data import load_split, split_video_names

    run_dir = sorted(REPO.glob(config_glob))[0]
    cfg = load_config(run_dir / "config.resolved.json")
    data, feature_schema = cfg["data"], cfg["feature_schema"]
    n_classes = int(cfg["model"]["num_classes"])
    train_feats, train_truths, id2name = load_split(data, data["split_train"], feature_schema=feature_schema)
    test_feats, test_truths, _ = load_split(data, data["split_eval"], feature_schema=feature_schema)
    names = split_video_names(data, data["split_eval"])
    labels = [id2name[i] for i in range(n_classes)]
    return train_feats, train_truths, test_feats, test_truths, names, labels, feature_schema


def evaluate(train_feats, train_truths, test_feats, test_truths, names, labels, indices) -> dict:
    """在给定通道子集上拟合 LDA 并逐视频评估（离线协议）。"""
    sys.path.insert(0, str(REPO / "tools"))
    from probe_split_shift import fit_multiclass_lda
    from framework.cleansight_eval.core.metrics import temporal_metrics

    x_train = np.concatenate([f[:, indices] for f in train_feats]).astype(np.float64)
    y_train = np.concatenate(train_truths)
    weights, bias = fit_multiclass_lda(x_train, y_train, len(labels))

    per_video = {}
    for name, frames, truth in zip(names, test_feats, test_truths):
        scores = frames[:, indices].astype(np.float64) @ weights.T + bias
        predicted = scores.argmax(axis=1)
        truth = np.asarray(truth)
        metrics = temporal_metrics({name: [labels[int(v)] for v in predicted]},
                                   {name: [labels[int(v)] for v in truth]}, labels)["segment"]
        record = {"edit": metrics["edit"] * 100,
                  "f1_01": metrics["f1_at_iou"]["0.10"] * 100,
                  "f1_025": metrics["f1_at_iou"]["0.25"] * 100}
        for index, label in enumerate(labels):
            mask = truth == index
            record[f"recall:{label}"] = float((predicted[mask] == index).mean() * 100) if mask.any() else None
        per_video[name] = record
    return per_video


def paired(base: dict, candidate: dict, key: str) -> dict | None:
    diffs = [candidate[v][key] - base[v][key] for v in base
             if v in candidate and candidate[v].get(key) is not None and base[v].get(key) is not None
             and abs(candidate[v][key] - base[v][key]) > 1e-9]
    if len(diffs) < 6:
        return None
    wins = sum(1 for d in diffs if d > 0)
    return {"n": len(diffs), "win": wins, "lose": len(diffs) - wins,
            "median": statistics.median(diffs), "p": float(wilcoxon(diffs).pvalue)}


def median_of(per_video: dict, key: str) -> float | None:
    values = [v[key] for v in per_video.values() if v.get(key) is not None]
    return statistics.median(values) if values else None


def channel_redundancy(train_feats, n_blocks: int, rows: int, cols: int, channels: int) -> dict:
    """通道语义之间的逐 (检测类, 区域) 相关性：|Pearson r| 的均值 / 最大 / 最小。

    冗余度高（r 接近 1）说明三类通道在"看同一件事"——此时把 144 维砍成 48 维几乎不丢信息。
    """
    matrix = np.concatenate(train_feats).astype(np.float64)
    pairs = {}
    for left in range(channels):
        for right in range(left + 1, channels):
            correlations = []
            for k in range(n_blocks):
                for r in range(rows * cols):
                    x = matrix[:, k * rows * cols * channels + r * channels + left]
                    y = matrix[:, k * rows * cols * channels + r * channels + right]
                    if x.std() < 1e-9 or y.std() < 1e-9:
                        continue  # 恒零通道无相关性可言
                    correlations.append(abs(float(np.corrcoef(x, y)[0, 1])))
            pairs[f"{CHANNEL_NAMES[left]}~{CHANNEL_NAMES[right]}"] = {
                "n_pairs": len(correlations),
                "mean_abs_r": float(np.mean(correlations)) if correlations else None,
                "max_abs_r": float(np.max(correlations)) if correlations else None,
                "min_abs_r": float(np.min(correlations)) if correlations else None,
            }
    return pairs


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    # 先验契约（不加载数据）：网格布局不成立的契约直接报错
    feature_schema, _model, layout = read_contract(args.config_run)
    if layout.get("groups"):
        raise SystemExit(
            f"本探针暂不支持**分组布局**契约（feature_layout.groups，如 actionmixed-roi-grid-v2）："
            f"version={feature_schema.get('version')}；请改用逐组块宽另行分析"
        )
    rows, cols = int(layout.get("rows", 2)), int(layout.get("cols", 3))
    channels = int(layout.get("channels", 3))
    block = rows * cols * channels
    dim = int(feature_schema["dim"])
    if dim % block:
        raise SystemExit(
            f"本探针只适用于**网格布局契约**（如 actionmixed-roi-grid-v1/v2）："
            f"dim={dim} 不能被块宽 rows*cols*channels={block} 整除（version={feature_schema.get('version')}）"
        )
    n_blocks = dim // block

    train_feats, train_truths, test_feats, test_truths, names, labels, _ = load_sequences(args.config_run)
    print(f"[probe] 契约 {feature_schema.get('version')}：{n_blocks} 检测类 × {rows * cols} 区域 × {channels} 通道 "
          f"= {dim} 维；动作类 {len(labels)} 个；test {len(names)} 个视频")

    subsets: dict[str, dict] = {"all": {}}
    for index, name in enumerate(CHANNEL_NAMES[:channels]):
        subsets[f"channel={name}"] = {"keep_channels": [index]}
    if channels >= 2:
        subsets["channel=presence+count"] = {"keep_channels": [0, 1]}
    for region in range(rows * cols):
        subsets[f"region={region}"] = {"keep_regions": [region]}

    results, per_video_by_subset = {}, {}
    print(f"\n{'子集':24} {'维数':>5} {'edit':>7} {'F1@.1':>7} {'F1@.25':>7}   edit 配对(vs all)")
    for name, spec in subsets.items():
        indices = channel_indices(n_blocks, rows, cols, channels, **spec)
        scored = evaluate(train_feats, train_truths, test_feats, test_truths, names, labels, indices)
        per_video_by_subset[name] = scored
        outcome = None if name == "all" else paired(per_video_by_subset["all"], scored, "edit")
        text = "—（基准）" if outcome is None else (
            f"n={outcome['n']} 胜/负={outcome['win']}/{outcome['lose']} 中位={outcome['median']:+6.2f} "
            f"p={outcome['p']:.4f}" + ("  <<< 显著" if outcome["p"] < 0.05 else ""))
        results[name] = {"dim": int(len(indices)),
                         "edit": median_of(scored, "edit"),
                         "f1_01": median_of(scored, "f1_01"),
                         "f1_025": median_of(scored, "f1_025"),
                         "per_class_recall": {label: median_of(scored, f"recall:{label}") or 0.0
                                              for label in labels},
                         "vs_all": outcome}
        print(f"{name:24} {len(indices):5} {results[name]['edit']:7.2f} {results[name]['f1_01']:7.2f} "
              f"{results[name]['f1_025']:7.2f}   {text}")

    print("\n逐类帧级 recall（%，行=类，列=子集）:")
    print(f"{'类':24}" + "".join(f"{name:>22}" for name in subsets))
    for label in labels:
        print(f"{label:24}" + "".join(f"{results[name]['per_class_recall'][label]:22.1f}" for name in subsets))

    redundancy = channel_redundancy(train_feats, n_blocks, rows, cols, channels)
    print("\n通道语义冗余度（逐 (检测类, 区域) 的 |Pearson r|，train split）:")
    for pair, stats in redundancy.items():
        if stats["mean_abs_r"] is None:
            print(f"  {pair:26} 无有效对照（恒零通道）")
        else:
            print(f"  {pair:26} 均值={stats['mean_abs_r']:.3f}  最大={stats['max_abs_r']:.3f}  "
                  f"最小={stats['min_abs_r']:.3f}  有效对照={stats['n_pairs']}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"contract": feature_schema.get("version"), "layout": layout,
                                    "subsets": results, "channel_redundancy": redundancy},
                                   ensure_ascii=False, indent=1))
        print(f"\n[probe] 结果已写入 {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
