"""端到端冒烟：MS-TCN 离线分割 train → eval → matrix（合成迷你 ActionMixed）。

机械验证全序列流水线全链路：逐帧全序列监督训练、fit_normalization 归一化统计写入 buffer
且随 checkpoint 存取复现、评估延迟标 N/A（离线不测实时延迟）、信封汇入异构矩阵。数值对齐
验收需在有真实数据的机器上执行。
"""

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from cleansight_eval.cli import eval as eval_cli
from cleansight_eval.cli import matrix as matrix_cli
from cleansight_eval.cli import train as train_cli
from cleansight_eval.core.envelope import MetricState

_ACTIONS = ["idle", "air_injection", "flush", "long_brush_insert", "long_brush_withdraw", "short_brush_cleaning"]


def _make_actionmixed(root, seed=0):
    """造迷你 ActionMixed：labels/<split>/<vid>.mp4.txt + frames/<split>/<vid>-<f>.txt。"""
    rng = np.random.default_rng(seed)
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "data.yaml").write_text(
        "nc: 6\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(_ACTIONS))
    )
    layout = {"train": {"v0": 60, "v1": 50}, "test": {"v2": 40}}
    for split, vids in layout.items():
        (root / "labels" / split).mkdir(parents=True)
        (root / "frames" / split).mkdir(parents=True)
        for vid, T in vids.items():
            frame_ids = list(range(1, T * 4, 4))
            actions = rng.integers(0, len(_ACTIONS), size=T)
            (root / "labels" / split / f"{vid}.mp4.txt").write_text(
                "\n".join(f"{fid} {a}" for fid, a in zip(frame_ids, actions)) + "\n"
            )
            for fid in frame_ids:
                n = rng.integers(0, 4)
                lines = [
                    f"{rng.integers(0,8)} {rng.random():.4f} {rng.random():.4f} {rng.random()*0.3:.4f} {rng.random()*0.3:.4f}"
                    for _ in range(n)
                ]
                (root / "frames" / split / f"{vid}.mp4-{fid:06d}.txt").write_text("\n".join(lines) + "\n")


def _write_config(path, data_root, model=None):
    cfg = {
        "pipeline": "full_sequence_temporal",
        "model": model or {"type": "mstcn", "input_dim": 40, "num_classes": 6, "hidden": 16},
        "data": {
            "name": "synthetic-actionmixed",
            "root": str(data_root),
            "action_mapping": "labels/data.yaml",
            "labels_dir": "labels",
            "frames_dir": "frames",
            "split_train": "train",
            "split_eval": "test",
        },
        "feature_schema": {"dim": 40, "version": "actionmixed-bbox-8cls-v1"},
        "train": {"epochs": 1, "lr": 0.01, "grad_clip": 5.0},
    }
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True))


def test_mstcn_end_to_end(tmp_path):
    data_root = tmp_path / "cleansight-ActionMixed"
    _make_actionmixed(data_root)
    cfg_path = tmp_path / "mstcn.yaml"
    _write_config(cfg_path, data_root)
    runs_dir = tmp_path / "runs"

    # train
    ckpt = train_cli.main(["--config", str(cfg_path), "--runs-dir", str(runs_dir)])
    assert ckpt.endswith(".pt")

    # 归一化统计已由 fit_normalization 写入 buffer 并随 checkpoint 持久化（非直通初值）。
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob.get("model_state", blob.get("state_dict", blob))
    assert "norm_mean" in state and "norm_std" in state
    assert not torch.allclose(state["norm_std"], torch.ones_like(state["norm_std"]))

    # eval → 一份信封：离线全序列，延迟标 N/A
    envelopes = eval_cli.main(["--config", str(cfg_path), "--ckpt", ckpt])
    assert len(envelopes) == 1
    data = json.loads(open(envelopes[0]).read())
    assert data["pipeline"] == "full_sequence_temporal"
    assert data["metrics"]["summary"]["acc"]["state"] in (
        MetricState.COMPUTED.value,
        MetricState.MISSING.value,
    )
    # 离线不测实时延迟：三态为 N/A（不是 0、不是缺失）
    assert data["performance"]["latency_mean_ms"]["state"] == MetricState.NOT_APPLICABLE.value
    assert data["inference"]["mode"] == "full_sequence"

    # matrix：mstcn 信封正常汇入
    matrix_json = matrix_cli.main(["--runs", str(runs_dir)])
    matrix = json.loads(open(matrix_json).read())
    assert len(matrix["rows"]) == 1


def test_mstcn2_end_to_end(tmp_path):
    """MS-TCN++（多 stage 深监督 + T-MSE）走 compute_loss 钩子的全链路冒烟。"""
    data_root = tmp_path / "cleansight-ActionMixed"
    _make_actionmixed(data_root)
    cfg_path = tmp_path / "mstcn2.yaml"
    _write_config(
        cfg_path,
        data_root,
        model={
            "type": "mstcn2",
            "input_dim": 40,
            "num_classes": 6,
            "hidden": 16,
            "num_stages": 2,
            "num_layers": 4,
        },
    )
    runs_dir = tmp_path / "runs"

    # train：走 compute_loss（多 stage + T-MSE），归一化 buffer 随 checkpoint 持久化。
    ckpt = train_cli.main(["--config", str(cfg_path), "--runs-dir", str(runs_dir)])
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob.get("model_state", blob.get("state_dict", blob))
    assert "norm_mean" in state and "norm_std" in state
    assert not torch.allclose(state["norm_std"], torch.ones_like(state["norm_std"]))
    # 多 stage：至少有 1 个精化 stage 的参数（refines.0.*）随权重存在。
    assert any(k.startswith("refines.0.") for k in state)

    # eval → 一份信封：离线全序列，延迟标 N/A，推理只取最后 stage。
    envelopes = eval_cli.main(["--config", str(cfg_path), "--ckpt", ckpt])
    data = json.loads(open(envelopes[0]).read())
    assert data["model"]["type"] == "mstcn2"
    assert data["performance"]["latency_mean_ms"]["state"] == MetricState.NOT_APPLICABLE.value
    assert data["inference"]["mode"] == "full_sequence"

    # 评估旁路自动出图：run 的 viz/ 下应有该 split 的分段条带图（按页切分，至少第一页）。
    run_dir = Path(ckpt).parents[1]  # <run>/checkpoints/<ckpt> → <run>
    assert (run_dir / "viz" / "segmentation-test-p01.png").exists()
