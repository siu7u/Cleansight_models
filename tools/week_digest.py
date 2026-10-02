"""短报告数字摘要：把「基线 vs 若干臂」的官方指标与配对检验渲染成可直接粘贴的短报告块。

用途
    周报 / 站会 / 群里同步只需要三五行的数字，但**手抄数字会过期**。本工具把"取数"这一步
    固化成一条命令：统计口径与配对检验**全部复用** ``tools/compare_runs.py``
    （``collect`` / ``median_of`` / ``paired``），本文件只做**渲染**，不新增任何统计口径。

三种长度（``--style``）
    brief  3 行：一行结论（最大 Δacc + 胜负 + p）、一行段级对照、一行来源
    short  约 12 行：再加各臂中位数表与配对要点表（默认）
    page   再加逐类（insert / withdraw / flush）与段结构读数

用法
    # 本周主对照（v4 vs v1，32 对 seed）
    PY=/home/caizh/programming/python_code/CleanSightBackend/.venv/bin/python
    $PY tools/week_digest.py \\
      --baseline "roi-grid-v1=runs/gpu-batch/recheck-flagship/h128/mstcn2-*,runs/acc-push/acc-roiv1-seedB/h128/mstcn2-*,runs/acc-push/acc-roiv1-seedC/h128/mstcn2-*" \\
      --arm "roi-grid-v4=runs/acc-push/acc-roiv4/h128/mstcn2-*,runs/acc-push/acc-roiv4-seedB/h128/mstcn2-*,runs/acc-push/acc-roiv4-seedC/h128/mstcn2-*" \\
      --style short

纪律
    输出**只有数字与口径**，不含结论性叙述——"所以该做什么"由人写（见
    ``docs/weeks/2026-09-26_2026-10-02/SHORT_REPORTS_HOWTO_20261001.md`` 的三层模板）。
    本工具不训练、不改产物，纯读 ``runs/`` 下的评估产物。
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_runs import collect, median_of, paired  # noqa: E402  复用统计口径

try:
    from scipy.stats import wilcoxon
except ImportError:  # scipy 缺失时退化为不报 p
    wilcoxon = None

# 渲染顺序：主口径在前后各一，段级与逐类跟在后面
HEADLINE = ("acc", "edit", "f1_025")
PER_CLASS = ("insert", "withdraw", "flush")
DETAIL = ("f1_01", "f1_05", "seg_ratio", "nonidle", "fp", "fn")


def parse_group(spec: str) -> tuple[str, list[str]]:
    """``标签=glob1,glob2`` → ``(标签, [glob...])``。"""

    if "=" not in spec:
        raise SystemExit(f"--arm/--baseline 需要 ``标签=glob[,glob...]`` 形式，收到 {spec!r}")
    label, globs = spec.split("=", 1)
    return label.strip(), [g.strip() for g in globs.split(",") if g.strip()]


def fmt(value, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _paired(arm_pv, base_pv, key):
    """``paired`` 在样本不足时只返回 ``{n, insufficient}``，这里统一成 None 便于渲染。"""

    r = paired(arm_pv, base_pv, key)
    return None if (r is None or r.get("insufficient")) else r


def seed_paired(arm_rows, base_rows, key):
    """**seed 级**配对（每行 = 一个 seed）：``acc`` 这类帧级 micro 指标没有逐视频记录，
    只能在这一层配对——与视频级配对不可混用，故输出里显式标注层级。"""

    a = {r["seed"]: r[key] for r in arm_rows if r.get(key) is not None}
    b = {r["seed"]: r[key] for r in base_rows if r.get(key) is not None}
    common = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in common if abs(a[s] - b[s]) > 1e-9]
    if not diffs:
        return None
    wins = sum(1 for d in diffs if d > 0)
    out = {"n": len(diffs), "win": wins, "lose": len(diffs) - wins,
           "median": float(statistics.median(diffs))}
    out["p"] = float(wilcoxon(diffs).pvalue) if (wilcoxon and len(diffs) >= 6) else float("nan")
    return out


def brief(baseline_label, base_rows, base_pv, arms) -> list[str]:
    """3 行版：结论 + 段级 + 来源。"""

    lines = []
    best = None
    for label, rows, pv in arms:
        r = seed_paired(rows, base_rows, "acc")
        if r and (best is None or r["median"] > best[1]["median"]):
            best = (label, r, rows)
    if best:
        label, r, rows = best
        lines.append(
            f"**{label} vs {baseline_label}**：acc 中位 {fmt(median_of(base_rows, 'acc'))} → "
            f"{fmt(median_of(rows, 'acc'))}（**seed 级**配对 Δ 中位 **{r['median']:+.2f}pp**，"
            f"胜/负 {r['win']}/{r['lose']}，p={r['p']:.2g}，n={r['n']}）")
        seg = seed_paired(rows, base_rows, "edit") or _paired(
            dict((lab, pv) for lab, _, pv in arms)[label], base_pv, "edit")
        if seg:
            lines.append(f"段级：edit Δ 中位 **{seg['median']:+.2f}pp**"
                         f"（{seg['win']}/{seg['lose']}，p={seg['p']:.2g}）")
    n_a = ", ".join(f"{lab} {len(rows)}" for lab, rows, _ in arms)
    lines.append(f"来源：基线 {baseline_label} {len(base_rows)} run；臂 {n_a}；"
                 f"口径=官方评测产物 seed 中位 + 逐 (seed,视频)×类配对")
    lines.append(f"_生成于 {time.strftime('%Y-%m-%d %H:%M')}（tools/week_digest.py，数字勿手改）_")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="短报告数字摘要（复用 compare_runs 口径）")
    ap.add_argument("--baseline", required=True, help="基线：``标签=glob[,glob...]``")
    ap.add_argument("--arm", action="append", required=True, help="臂：``标签=glob[,...]``，可多次")
    ap.add_argument("--style", choices=("brief", "short", "page"), default="short")
    ap.add_argument("--title", default=None, help="报告标题（缺省自动生成）")
    args = ap.parse_args(argv)

    b_label, b_globs = parse_group(args.baseline)
    b_rows, b_pv = collect(b_globs)
    if not b_rows:
        raise SystemExit(f"基线 {b_label} 没匹配到任何 run（检查 glob）")
    arm_specs = [parse_group(s) for s in args.arm]

    arms = []
    for label, globs in arm_specs:
        rows, pv = collect(globs)
        if not rows:
            print(f"⚠️  跳过 {label}：没匹配到 run", file=sys.stderr)
            continue
        arms.append((label, rows, pv))
    if not arms:
        raise SystemExit("没有任何臂匹配到 run")

    out = [f"## {args.title or '数字摘要'}", ""]
    out += brief(b_label, b_rows, b_pv, arms)

    if args.style in ("short", "page"):
        out += ["", "| 指标 | " + " | ".join([b_label] + [lab for lab, _, _ in arms]) + " |",
                "|---|" + "---:|" * (1 + len(arms))]
        for key in HEADLINE + (DETAIL if args.style == "page" else ()):
            cells = [fmt(median_of(b_rows, key))] + [fmt(median_of(rows, key)) for _, rows, _ in arms]
            out.append(f"| {key} | " + " | ".join(cells) + " |")
        out += ["", "**seed 级配对**（每行 = 一个 seed；`acc` 这类帧级 micro 指标只在此层可配对）", "",
                "| 臂 | 指标 | n | 胜/负 | Δ 中位 | p |", "|---|---|---:|---|---:|---:|"]
        for label, rows, _ in arms:
            for key in ("acc", "edit", "f1_025"):
                r = seed_paired(rows, b_rows, key)
                if not r:
                    out.append(f"| {label} | {key} | 0 | — | — | 样本不足 |")
                    continue
                flag = " ✅" if r["p"] < 0.05 else ""
                out.append(f"| {label} | {key} | {r['n']} | {r['win']}/{r['lose']} | "
                           f"{r['median']:+.2f} | {r['p']:.2g}{flag} |")
        out += ["", "**视频级配对**（逐 (seed,视频)×类 Wilcoxon，差值 = 臂 − 基线）", "",
                "| 臂 | 指标 | n | 胜/负 | Δ 中位 | p |", "|---|---|---:|---|---:|---:|"]
        # 视频级表不含 acc：帧级 micro 指标在产物里没有逐视频记录，配对样本恒为 0
        keys = ("edit", "f1_025") + (PER_CLASS if args.style == "page" else ())
        for label, rows, pv in arms:
            for key in keys:
                r = _paired(pv, b_pv, key)
                if not r:
                    out.append(f"| {label} | {key} | 0 | — | — | 样本不足 |")
                    continue
                flag = " ✅" if r["p"] < 0.05 else ""
                out.append(f"| {label} | {key} | {r['n']} | {r['win']}/{r['lose']} | "
                           f"{r['median']:+.2f} | {r['p']:.2g}{flag} |")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
