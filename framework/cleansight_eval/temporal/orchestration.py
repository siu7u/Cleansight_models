"""时序纵编排（TemporalOrchestrator）。

时序纵自持训练/评估主体：forward/loss 循环、指标口径、喂入语义。CLI 只按
``cfg["task"]`` 分派到本编排器。时序专属的配置校验（feature_schema / input_dim /
num_classes）也在本纵内完成，core 不再强制这些字段（§4.2）。

本编排器与 detection 纵**不共享**任何 family/feeding/task 抽象；两纵仅在 core 的
信封与矩阵处汇合。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:  # tqdm 可选，缺失时退化为原样迭代
    def tqdm(iterable, **_kwargs):
        return iterable

from ..core.checkpoint import load_checkpoint, save_checkpoint
from ..core.environment import now_stamp, set_seed
from ..core.envelope import EvalEnvelope
from ..core.integrity import check_envelope_complete, check_feature_schema
from ..core.run import RunContext
from .feeding import get_feeding
from .perf import measure_single_tick, not_applicable_perf
from .family import get_family
from .loader import load_split
from .metrics import compute_temporal_metrics
from .util import compute_class_weights


class TemporalOrchestrator:
    task_id = "temporal"

    def validate_config(self, cfg: dict) -> None:
        for k in ("input_dim", "num_classes"):
            if k not in cfg.get("model", {}):
                raise ValueError(f"时序任务 model 缺少必要字段: {k}")
        if "feature_schema" not in cfg:
            raise ValueError("时序任务需要 feature_schema（用于训练前的特征兼容检查）")
        if "train" not in cfg:
            raise ValueError("时序任务需要 train 段（epochs/lr/batch_size/window）")
        data = cfg.get("data", {})
        for k in ("root", "split_train", "split_eval"):
            if k not in data:
                raise ValueError(f"时序任务 data 段缺少必要字段: {k}（用数据集内建目录切分）")
        # 喂入模式（训练与评估共用同一个）必须已注册，且能用于训练。
        feeding = get_feeding(cfg["feeding"])
        if not hasattr(feeding, "build_datasets"):
            raise ValueError(f"喂入模式 {feeding.name} 不能用于时序训练（未实现 build_datasets）")

    def train(self, cfg: dict, runs_dir: str, seed: int, device) -> str:
        train_cfg = cfg["train"]
        window = train_cfg.get("window", 64)

        set_seed(seed)
        family = get_family(cfg["family"])
        model = family.build_network(cfg["model"]).to(device)

        run = RunContext(runs_dir, family=cfg["family"])
        run.save_config(cfg)
        run.save_env(device, seed=seed)

        # data loader（features 契约：读 ActionMixed + bbox→40维）+ 训练前 schema 兼容检查（§7.3）
        features, truths, _ = load_split(cfg["data"], cfg["data"]["split_train"], window=window)
        problems = check_feature_schema(features[0].shape[1], cfg.get("feature_schema"))
        if problems:
            raise ValueError("特征 schema 与配置不兼容:\n  - " + "\n  - ".join(problems))

        # 训练前钩子（统一契约）：让 family 有机会按训练数据准备（如离线分割 fit 归一化统计）。
        # GRU 等无需归一化的族为空操作——编排器无条件调用，不按模型类型分支。
        family.prepare(model, features)

        # 本实验的喂入模式（训练与评估共用），由它构造样本容器（窗口/末帧规格的单一真源）。
        # build_datasets 是训练与评估共用的核心构造；训练侧再 ConcatDataset 拼接成一个可批处理集。
        # batch 策略由喂入模式决定：全序列变长→逐条(1)，窗口样本→用 cfg 的 batch_size。
        feeding = get_feeding(cfg["feeding"])
        train_ds = ConcatDataset(feeding.build_datasets(features, truths, list(range(len(features))), window))
        batch_size = feeding.train_batch_size or train_cfg.get("batch_size", 32)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        weights = compute_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([weights[i] for i in sorted(weights)], dtype=torch.float32).to(device)
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )
        grad_clip = train_cfg.get("grad_clip")  # 值驱动（非模型分支）：缺省则不裁剪

        model.train()
        epochs = train_cfg.get("epochs", 20)
        for _epoch in tqdm(range(1, epochs + 1), desc="train"):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits = family.forward(model, x)
                loss = family.compute_loss(logits, y, criterion)
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        ckpt_path = run.checkpoints_dir / f"{cfg['family']}-final-{now_stamp()}.pt"
        meta = family.checkpoint_meta(
            cfg["model"],
            cfg.get("feature_schema", {}),
            extra={
                "task": cfg["task"],
                "feeding": cfg["feeding"],  # 记录本模型是怎么喂训练的（评估须同模式）
                "window": window,
                "num_params": sum(p.numel() for p in model.parameters()),
                "trained_at": now_stamp(),
                "train": train_cfg,
            },
        )
        save_checkpoint(ckpt_path, model.state_dict(), meta)
        print(f"[train] run_dir={run.dir}")
        print(f"[train] checkpoint={ckpt_path}")
        return str(ckpt_path)

    def evaluate(self, cfg: dict, ckpt: str, feeding_name: str, device) -> EvalEnvelope:
        # 期望配置来自实验配置的 model 段；错配立即抛错（§8.1）。
        expected = {"family": cfg["family"], **{k: cfg["model"][k] for k in ("input_dim", "num_classes")}}
        state_dict, meta = load_checkpoint(ckpt, expected=expected, map_location=device)

        family = get_family(meta["family"])
        model = family.build_network(meta["model"]).to(device)
        model.load_state_dict(state_dict)

        window = meta.get("window", cfg["train"].get("window", 64))
        features, truths, id2name = load_split(cfg["data"], cfg["data"]["split_eval"], window=window)
        idx_to_action = id2name  # {action_id: name}

        # 评估集与训练集共用同一核心构造（feeding.build_datasets）：各喂入模式自持样本形态，
        # 编排器不再硬编码窗口容器（消除全序列寄生 EndoDataset 的巧合）。
        mode = get_feeding(feeding_name)
        datasets = mode.build_datasets(features, truths, list(range(len(features))), window)
        result = mode.evaluate(family, model, datasets, device)

        all_preds = np.concatenate(result.video_preds)
        all_gts = np.concatenate(result.video_gts)
        pred_labels = [idx_to_action[p] for p in all_preds]
        gt_labels = [idx_to_action[g] for g in all_gts]
        metrics = compute_temporal_metrics(pred_labels, gt_labels)

        if mode.requires_performance:
            performance = measure_single_tick(family, model, window, cfg["model"]["input_dim"], device)
        else:
            performance = not_applicable_perf()

        envelope = EvalEnvelope(
            family=meta["family"],
            model_id=f"{meta['family']}-{cfg['model'].get('hidden', '')}h",
            task=cfg["task"],
            feeding=feeding_name,
            checkpoint=str(ckpt),
            dataset=cfg["data"].get("name", cfg["data"].get("root")),
            feature_schema=meta.get("feature_schema", cfg.get("feature_schema", {})),
            metrics=metrics,
            performance=performance,
            feeding_semantics=result.semantics,
            num_params=meta.get("num_params"),
            timestamp=now_stamp(),
        )
        envelope.integrity = check_envelope_complete(envelope)
        return envelope
