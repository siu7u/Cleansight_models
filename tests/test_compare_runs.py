"""tools/compare_runs.py + tools/probe_offline_postprocess.py：配对对照与后处理扫描的单元测试。

验收标准（verification-first）：
- `compare_runs` 的官方汇总取 seed 中位数、逐视频重算值与 `temporal_metrics` 同口径；
- 配对检验按 (seed, 视频) 配对，差值方向 = left − right，零差值对剔除，n<6 报样本不足；
- 构造一组"左侧系统性更好"的合成 run → 配对检验必须显著（胜/负与 p 同向）；
- `merge_short` 把短段并入较长邻段；`median_filter` 做多数类滤波且长度不变；
- 后处理扫描能报出"平滑对抖动预测加分、对干净预测无感"的方向（合成两臂验证）。

全部用合成产物（tmp_path），不依赖真实数据集与 checkpoint。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.probe_offline_postprocess import median_filter, merge_short, score, aggregate  # noqa: E402
from tools import compare_runs  # noqa: E402

LABELS = ["idle", "flush", "insert"]


def write_run(root: Path, name: str, seed: int, predicted: dict[str, list[str]], truth: dict[str, list[str]],
              edit: float, f1_01: float, per_class: dict | None = None) -> Path:
    """写一个最小可用的 run 目录：env.json + evals/*.evaluation.json + artifacts/*.predictions.json。"""
    run = root / name
    (run / "evals").mkdir(parents=True)
    (run / "artifacts").mkdir(parents=True)
    (run / "env.json").write_text(json.dumps({"seed": seed}))
    (run / "evals" / "full_sequence_temporal-x.evaluation.json").write_text(json.dumps({
        "metrics": {
            "summary": {
                "edit": {"state": "computed", "value": edit},
                "f1@0.1": {"state": "computed", "value": f1_01},
                "f1@0.25": {"state": "computed", "value": f1_01 / 2},
                "acc": {"state": "computed", "value": 50.0},
                "tp@0.5": {"state": "computed", "value": 1},
                "fp@0.5": {"state": "computed", "value": 2},
                "fn@0.5": {"state": "computed", "value": 3},
            },
            "details": {"temporal": {"frame": {"per_class": per_class or {}}}},
        },
    }))
    (run / "artifacts" / "full_sequence_temporal-x.predictions.json").write_text(json.dumps({
        "labels": [{"name": name} for name in LABELS],
        "items": {video: {"predicted_labels": predicted[video], "truth_labels": truth[video]}
                  for video in predicted},
    }))
    return run


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    """把仓库根切到 tmp_path，使相对 glob 直接可用（工具只接受相对仓库根的模式）。"""
    monkeypatch.setattr(compare_runs, "REPO", tmp_path)
    truth = {f"v{i}": ["idle"] * 8 + ["flush"] * 6 + ["idle"] * 6 for i in range(6)}
    clean = {video: list(seq) for video, seq in truth.items()}
    jittery = {}
    for video, seq in truth.items():
        noisy = list(seq)
        for position in range(2, 18, 4):
            noisy[position] = "insert"
        jittery[video] = noisy
    for seed in (1, 2):
        write_run(tmp_path, f"clean-{seed}", seed, clean, truth, 60.0 + seed, 50.0,
                  per_class={"idle": {"support": 20, "precision": 0.9, "recall": 0.8, "f1": 0.85},
                             "flush": {"support": 6, "precision": 0.5, "recall": 0.4, "f1": 0.44}})
        write_run(tmp_path, f"jittery-{seed}", seed, jittery, truth, 20.0 + seed, 20.0,
                  per_class={"idle": {"support": 20, "precision": 0.6, "recall": 0.5, "f1": 0.55},
                             "flush": {"support": 6, "precision": 0.2, "recall": 0.1, "f1": 0.13}})
    return tmp_path


def test_collect_takes_official_summary_and_recomputes_per_video(corpus):
    rows, per_video = compare_runs.collect(["clean-*"])
    assert sorted(row["seed"] for row in rows) == [1, 2]
    assert compare_runs.median_of(rows, "edit") == pytest.approx(61.5)  # 官方 summary 中位数
    assert set(per_video[1]) == {f"v{i}" for i in range(6)}
    # 逐视频重算：干净预测 = 真值 → edit 100
    assert per_video[1]["v0"]["edit"] == pytest.approx(100.0)
    assert per_video[1]["v0"]["f1_01"] == pytest.approx(100.0)
    # 抖动预测：噪声帧落在 flush 段内（下标 10）→ flush 召回 5/6；且没有真值 insert 帧（不可评估）
    _, jittery = compare_runs.collect(["jittery-*"])
    assert jittery[1]["v0"]["flush"] == pytest.approx(500.0 / 6)
    assert jittery[1]["v0"]["long_brush_insert"] is None


def test_paired_reports_direction_and_significance(corpus):
    left_rows, left_pairs = compare_runs.collect(["clean-*"])
    right_rows, right_pairs = compare_runs.collect(["jittery-*"])
    outcome = compare_runs.paired(left_pairs, right_pairs, "edit")
    assert outcome["n"] == 12 and outcome["win"] == 12 and outcome["lose"] == 0
    assert outcome["median"] > 0 and outcome["p"] < 0.05  # 左侧系统性更好 → 显著


def test_paired_insufficient_sample_is_flagged(corpus):
    _, left_pairs = compare_runs.collect(["clean-*"])
    outcome = compare_runs.paired(left_pairs, left_pairs, "edit")
    assert outcome["insufficient"] is True and outcome["n"] == 0  # 同源 → 差值全 0 → 无有效对


def test_cli_prints_table_and_writes_json(corpus, capsys):
    exit_code = compare_runs.main(["--left", "clean-*", "--left-label", "clean",
                                   "--right", "jittery-*", "--right-label", "jittery",
                                   "--metrics", "edit,f1_01", "--json", str(corpus / "cmp.json")])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "配对检验" in output and "edit" in output
    payload = json.loads((corpus / "cmp.json").read_text())
    assert payload["left"]["n_runs"] == 2 and payload["left"]["medians"]["edit"] == pytest.approx(61.5)
    assert payload["paired"]["edit"]["win"] == 12


def test_per_class_pooled_metrics_and_precision_metric(corpus, capsys):
    """`--per-class` 读官方 details 的 micro-pool 逐类指标；`类:precision` 走逐视频配对。"""

    rows, per_video = compare_runs.collect(["clean-*"])
    assert rows[0]["_pooled"]["flush"]["precision"] == pytest.approx(50.0)   # 0.5 → 百分数
    assert rows[0]["_pooled"]["flush"]["recall"] == pytest.approx(40.0)
    assert rows[0]["_pooled"]["idle"]["support"] == 20

    # 逐视频类 precision：干净预测 == 真值 → 该类 precision 100
    assert per_video[1]["v0"]["flush:precision"] == pytest.approx(100.0)

    exit_code = compare_runs.main(["--left", "clean-*", "--left-label", "clean",
                                   "--right", "jittery-*", "--right-label", "jittery",
                                   "--metrics", "edit,insert,flush:precision", "--per-class"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "逐类帧级指标" in output and "idle (n=20)" in output
    assert "P / R / F1" in output
    # flush:precision：干净一侧 100、抖动一侧更低 → 左侧胜
    assert "flush:precision" in output


def test_median_filter_is_majority_and_keeps_length():
    sequence = np.array([0, 0, 1, 0, 0, 0])
    filtered = median_filter(sequence, 3)
    assert len(filtered) == len(sequence)
    assert list(filtered) == [0, 0, 0, 0, 0, 0]  # 孤立的 1 被邻域多数抹掉
    assert list(median_filter(sequence, 1)) == list(sequence)


def test_merge_short_absorbs_short_segments_into_longer_neighbour():
    assert list(merge_short(np.array([0, 0, 0, 1, 0, 0, 0]), 3)) == [0] * 7
    # 短段两侧等长 → 并入左邻（实现约定），长段不被破坏
    assert list(merge_short(np.array([1, 1, 1, 2, 2, 2]), 3)) == [1, 1, 1, 2, 2, 2]
    assert list(merge_short(np.array([2, 2, 2, 1, 2, 2, 2]), 2)) == [2, 2, 2, 2, 2, 2, 2]


def test_postprocess_helps_jittery_but_not_clean_predictions(corpus):
    """合成的"抖动 vs 干净"两臂：合并对抖动预测加分、对干净预测不改变。"""
    truth = {f"v{i}": ["idle"] * 8 + ["flush"] * 6 + ["idle"] * 6 for i in range(6)}
    labels = LABELS

    def build(predicted):
        runs = {1: {video: (np.array([labels.index(v) for v in predicted[video]]),
                            np.array([labels.index(v) for v in truth[video]])) for video in truth}}
        return runs

    jittery = {}
    clean = {}
    for video, seq in truth.items():
        noisy = list(seq)
        for position in range(2, 18, 4):
            noisy[position] = "insert"
        jittery[video] = noisy
        clean[video] = list(seq)

    jittery_base = aggregate(score(build(jittery), labels, lambda s: s))
    jittery_merged = aggregate(score(build(jittery), labels, lambda s: merge_short(s, 3)))
    clean_base = aggregate(score(build(clean), labels, lambda s: s))
    clean_merged = aggregate(score(build(clean), labels, lambda s: merge_short(s, 3)))

    assert jittery_merged["edit"] > jittery_base["edit"]
    assert jittery_merged["segments"] < jittery_base["segments"]
    assert clean_merged["edit"] == pytest.approx(clean_base["edit"]) == pytest.approx(100.0)


def test_collect_from_variant_eval_dir_pairs_predictions_by_stem(corpus):
    """`--left-eval-dir`：评估读口径变体目录，预测按**同名 stem** 从 run 目录的 artifacts 取。"""
    root = corpus
    variant = root / "_eval_md7" / "clean-1"
    variant.mkdir(parents=True)
    stem = "full_sequence_temporal-x"
    # 变体评估：与 run 目录里的预测同 stem → 必须能配上
    (variant / f"{stem}.evaluation.json").write_text(json.dumps({
        "metrics": {"summary": {"edit": {"state": "computed", "value": 99.0},
                                "f1@0.1": {"state": "computed", "value": 88.0},
                                "acc": {"state": "computed", "value": 77.0}},
                    "details": {"temporal": {"frame": {"per_class": {}}}}},
    }))

    rows, per_video = compare_runs.collect(["clean-*"], ["_eval_md7/clean-*"])

    assert [r["seed"] for r in rows] == [1]
    assert rows[0]["edit"] == pytest.approx(99.0)   # 官方口径来自变体评估
    assert per_video[1]["v0"]["edit"] == pytest.approx(100.0)  # 逐视频重算来自同名 stem 的预测

    # stem 对不上时不得静默用别的预测充当：应直接跳过该 run
    (variant / "other.predictions.json").write_text("{}")
    (variant / "mismatch.evaluation.json").write_text((variant / f"{stem}.evaluation.json").read_text())
    rows2, _ = compare_runs.collect(["clean-*"], ["_eval_md7/clean-*"])
    assert len(rows2) == 1  # 仍然只认能配上预测的那条
