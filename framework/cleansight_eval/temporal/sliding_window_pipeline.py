"""历史滑窗时序流水线（有界因果窗流式推理）。

一条完整的训练+评估单元：逐 tick 只喂当前窗口 ``[1, window, F]``、取窗口**最后一帧** logits、
**末帧监督**；评估逐窗推理，经因果平滑 ``causal_decision`` 得稳定预测。前 ``window-1`` 帧为
冷启动填充 idle，每个视频开始前 reset。训练与评估使用同一种数据组织（窗口 + 末帧标签），
按需记录单 tick 延迟。

模型作为可替换组件（GRU / causal-TCN…）由 ``model.type`` 选取，**必须因果**（``is_causal``
为真），否则拒绝——非因果模型（如 MS-TCN）逐窗末帧语义不成立。数据读取/指标/延迟工具与
全序列流水线共享（``data`` / ``metrics``），但绝不跨到 detection 域。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # tqdm 可选，缺失时退化为原样迭代
    def tqdm(iterable, **_kwargs):
        return iterable

from ..core.checkpoint import load_checkpoint, save_checkpoint
from ..core.environment import now_stamp, set_seed
from ..core.envelope import EvalEnvelope, format_params
from ..core.integrity import check_envelope_complete, check_feature_schema
from ..core.run import RunContext
from .data import build_temporal_meta, load_split
from .metrics import causal_decision, compute_temporal_metrics, measure_single_tick
from .models import build_model, is_causal
from .util import compute_class_weights

IDLE_ID = 0
MIN_DURATION = 25  # causal_decision 内部最小持续时长，此处仅用于语义描述


class SlidingWindowDataset(Dataset):
    """因果滑窗数据集：样本为 ``[window, F]``，标签为窗口最后一帧类别。"""

    def __init__(self, features: np.ndarray, labels, window: int):
        self.x = torch.from_numpy(features).float()
        self.y = torch.tensor(labels, dtype=torch.long)
        self.w = window

    def __len__(self):
        return len(self.x) - self.w + 1

    def __getitem__(self, idx):
        x = self.x[idx : idx + self.w]
        y = self.y[idx + self.w - 1]
        return x, y


class SlidingWindowTemporalPipeline:
    pipeline_name = "sliding_window_temporal"

    def validate_config(self, cfg: dict) -> None:
        model = cfg.get("model", {})
        for k in ("type", "input_dim", "num_classes"):
            if k not in model:
                raise ValueError(f"滑窗时序流水线 model 缺少必要字段: {k}")
        if not is_causal(model["type"]):
            raise ValueError(
                f"滑窗流水线要求因果模型（逐窗取末帧），{model['type']!r} 非因果，不可用于滑窗"
            )
        if "feature_schema" not in cfg:
            raise ValueError("时序流水线需要 feature_schema（用于训练前的特征兼容检查）")
        if "train" not in cfg:
            raise ValueError("时序流水线需要 train 段（epochs/lr/batch_size/window）")
        data = cfg.get("data", {})
        for k in ("root", "split_train", "split_eval"):
            if k not in data:
                raise ValueError(f"时序流水线 data 段缺少必要字段: {k}（用数据集内建目录切分）")

    def train(self, cfg: dict, runs_dir: str, seed: int, device) -> str:
        train_cfg = cfg["train"]
        model_cfg = cfg["model"]
        window = train_cfg.get("window", 64)

        set_seed(seed)
        model = build_model(model_cfg).to(device)

        run = RunContext(runs_dir, label=model_cfg["type"])
        run.save_config(cfg)
        run.save_env(device, seed=seed)

        features, truths, _ = load_split(cfg["data"], cfg["data"]["split_train"], window=window)
        problems = check_feature_schema(features[0].shape[1], cfg.get("feature_schema"))
        if problems:
            raise ValueError("特征 schema 与配置不兼容:\n  - " + "\n  - ".join(problems))

        if hasattr(model, "fit_normalization"):
            model.fit_normalization(features)

        # 窗口 + 末帧标签：窗口样本可批处理，用 cfg 的 batch_size。
        train_ds = ConcatDataset(
            [SlidingWindowDataset(features[i], truths[i], window) for i in range(len(features))]
        )
        train_loader = DataLoader(train_ds, batch_size=train_cfg.get("batch_size", 32), shuffle=True)

        weights = compute_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([weights[i] for i in sorted(weights)], dtype=torch.float32).to(device)
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
        grad_clip = train_cfg.get("grad_clip")

        model.train()
        epochs = train_cfg.get("epochs", 20)
        for _epoch in tqdm(range(1, epochs + 1), desc="train"):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)  # [B, window, C]
                # 因果契约：只对窗口最后一帧计算损失。
                loss = criterion(logits[:, -1, :], y)
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        ckpt_path = run.checkpoints_dir / f"{model_cfg['type']}-final-{now_stamp()}.pt"
        extra = {"normalizer": "zscore/train-set/buffers/v1"} if hasattr(model, "fit_normalization") else None
        meta = build_temporal_meta(
            model_cfg,
            cfg.get("feature_schema", {}),
            pipeline=self.pipeline_name,
            window=window,
            num_params=sum(p.numel() for p in model.parameters()),
            train_cfg=train_cfg,
            trained_at=now_stamp(),
            extra=extra,
        )
        save_checkpoint(ckpt_path, model.state_dict(), meta)
        print(f"[train] run_dir={run.dir}")
        print(f"[train] checkpoint={ckpt_path}")
        return str(ckpt_path)

    def evaluate(self, cfg: dict, ckpt: str, device) -> EvalEnvelope:
        model_cfg = cfg["model"]
        expected = {"type": model_cfg["type"], "input_dim": model_cfg["input_dim"], "num_classes": model_cfg["num_classes"]}
        state_dict, meta = load_checkpoint(ckpt, expected=expected, map_location=device)

        model = build_model(meta["model"]).to(device)
        model.load_state_dict(state_dict)

        window = meta.get("window") or cfg["train"].get("window", 64)
        features, truths, id2name = load_split(cfg["data"], cfg["data"]["split_eval"], window=window)
        datasets = [SlidingWindowDataset(features[i], truths[i], window) for i in range(len(features))]

        model.eval()
        video_preds, video_gts = [], []
        with torch.no_grad():
            for ds in datasets:
                total_frames = ds.x.shape[0]
                preds = np.zeros(total_frames, dtype=np.int64)
                preds[: window - 1] = IDLE_ID  # 冷启动

                pending, stable, count = None, IDLE_ID, 0  # 每视频 reset
                for i in range(len(ds)):
                    x, _ = ds[i]
                    x = x.unsqueeze(0).to(device)  # [1, window, F]
                    last = model(x)[0, -1]  # 末帧 logits
                    pending, stable, count = causal_decision(last, pending, stable, count)
                    preds[i + window - 1] = stable
                video_preds.append(preds)
                video_gts.append(ds.y.numpy())

        all_preds = np.concatenate(video_preds)
        all_gts = np.concatenate(video_gts)
        pred_labels = [id2name[p] for p in all_preds]
        gt_labels = [id2name[g] for g in all_gts]
        metrics = compute_temporal_metrics(pred_labels, gt_labels)

        performance = measure_single_tick(model, window, model_cfg["input_dim"], device)
        semantics = {
            "mode": "windowed_causal",
            "sees": "causal_sliding_window",
            "window": window,
            "advance": 1,
            "cold_start": f"前 {window - 1} 帧填充 idle",
            "reset": "per_video",
            "smoothing": f"causal_decision(min_duration={MIN_DURATION})",
        }

        envelope = EvalEnvelope(
            model_type=meta["type"],
            model_id=f"{meta['type']}-{format_params(meta.get('num_params'))}",
            pipeline=self.pipeline_name,
            checkpoint=str(ckpt),
            dataset=cfg["data"].get("name", cfg["data"].get("root")),
            feature_schema=meta.get("feature_schema", cfg.get("feature_schema", {})),
            metrics=metrics,
            performance=performance,
            inference_semantics=semantics,
            num_params=meta.get("num_params"),
            timestamp=now_stamp(),
        )
        envelope.integrity = check_envelope_complete(envelope)
        return envelope
