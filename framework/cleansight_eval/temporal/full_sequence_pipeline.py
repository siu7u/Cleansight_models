"""全序列时序流水线（离线全量推理）。

一条完整的训练+评估单元：一次看到完整特征序列 ``[1, T, F]``、**逐帧监督**、逐帧 argmax
推理。训练与评估使用同一种数据组织（整段序列 + 逐帧标签），不测实时延迟（离线，标 N/A）。

模型作为可替换组件（GRU / MS-TCN…）由 ``model.type`` 选取，只需满足 ``[B,T,F] -> [B,T,C]``
的前向约定；监督口径与推理方式由本流水线拥有，不写在模型里。数据读取/指标/延迟工具与
滑窗流水线共享（``data`` / ``metrics``），但绝不跨到 detection 域。
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
from .metrics import compute_temporal_metrics, not_applicable_perf
from .models import build_model
from .util import compute_class_weights

FULL_SEQUENCE_SEMANTICS = {
    "mode": "full_sequence",
    "sees": "full_sequence",
    "windowing": "none",
    "reset": "per_video",
    "note": "全量一次推理，不代表实时行为",
}


class FullSequenceDataset(Dataset):
    """整段序列样本：一条视频一个样本，``x=[T, F]``、``y=[T]``（逐帧标签）。

    变长 T 由 ``batch_size=1`` 保证不触发默认 collate 的批内对齐。
    """

    def __init__(self, features: np.ndarray, labels):
        self.x = torch.from_numpy(np.asarray(features)).float()
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.x, self.y


class FullSequenceTemporalPipeline:
    pipeline_name = "full_sequence_temporal"

    def validate_config(self, cfg: dict) -> None:
        model = cfg.get("model", {})
        for k in ("type", "input_dim", "num_classes"):
            if k not in model:
                raise ValueError(f"全序列时序流水线 model 缺少必要字段: {k}")
        if "feature_schema" not in cfg:
            raise ValueError("时序流水线需要 feature_schema（用于训练前的特征兼容检查）")
        if "train" not in cfg:
            raise ValueError("时序流水线需要 train 段（epochs/lr/batch_size）")
        data = cfg.get("data", {})
        for k in ("root", "split_train", "split_eval"):
            if k not in data:
                raise ValueError(f"时序流水线 data 段缺少必要字段: {k}（用数据集内建目录切分）")

    def train(self, cfg: dict, runs_dir: str, seed: int, device) -> str:
        train_cfg = cfg["train"]
        model_cfg = cfg["model"]

        set_seed(seed)
        model = build_model(model_cfg).to(device)

        run = RunContext(runs_dir, label=model_cfg["type"])
        run.save_config(cfg)
        run.save_env(device, seed=seed)

        # data（features 契约：读 ActionMixed + bbox→40维）+ 训练前 schema 兼容检查。
        features, truths, _ = load_split(cfg["data"], cfg["data"]["split_train"])
        problems = check_feature_schema(features[0].shape[1], cfg.get("feature_schema"))
        if problems:
            raise ValueError("特征 schema 与配置不兼容:\n  - " + "\n  - ".join(problems))

        # 可选归一化钩子（duck-type）：模型若自带 fit_normalization（如 MS-TCN）则按训练集统计写入。
        # buffer 随 state_dict 存取，评估时自动应用同一归一化。
        if hasattr(model, "fit_normalization"):
            model.fit_normalization(features)

        # 整段 + 逐帧：变长全序列逐条喂入（batch_size=1，不做批内对齐）。
        train_ds = ConcatDataset(
            [FullSequenceDataset(features[i], truths[i]) for i in range(len(features))]
        )
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

        weights = compute_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([weights[i] for i in sorted(weights)], dtype=torch.float32).to(device)
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
        grad_clip = train_cfg.get("grad_clip")  # 值驱动：缺省则不裁剪

        model.train()
        epochs = train_cfg.get("epochs", 20)
        for _epoch in tqdm(range(1, epochs + 1), desc="train"):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)  # [B, T, C]
                # 逐帧监督：整段序列每帧都算 CE（与滑窗末帧监督相对）。
                loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
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
            window=None,  # 全序列不加窗
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

        features, truths, id2name = load_split(cfg["data"], cfg["data"]["split_eval"])

        model.eval()
        video_preds, video_gts = [], []
        with torch.no_grad():
            for feats, gt in zip(features, truths):
                x = torch.from_numpy(feats).float().unsqueeze(0).to(device)  # [1, T, F]
                logits = model(x)  # [1, T, C]
                preds = torch.argmax(logits[0], dim=-1).cpu().numpy().astype(np.int64)
                video_preds.append(preds)
                video_gts.append(gt)

        all_preds = np.concatenate(video_preds)
        all_gts = np.concatenate(video_gts)
        pred_labels = [id2name[p] for p in all_preds]
        gt_labels = [id2name[g] for g in all_gts]
        metrics = compute_temporal_metrics(pred_labels, gt_labels)

        envelope = EvalEnvelope(
            model_type=meta["type"],
            model_id=f"{meta['type']}-{format_params(meta.get('num_params'))}",
            pipeline=self.pipeline_name,
            checkpoint=str(ckpt),
            dataset=cfg["data"].get("name", cfg["data"].get("root")),
            feature_schema=meta.get("feature_schema", cfg.get("feature_schema", {})),
            metrics=metrics,
            performance=not_applicable_perf("离线全序列不测实时延迟"),
            inference_semantics=FULL_SEQUENCE_SEMANTICS,
            num_params=meta.get("num_params"),
            timestamp=now_stamp(),
        )
        envelope.integrity = check_envelope_complete(envelope)
        return envelope
