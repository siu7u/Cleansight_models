"""全序列时序流水线（离线全量推理）。

一条完整的训练+推理单元：一次看到完整特征序列 ``[1, T, F]``、**逐帧监督**、逐帧 argmax
推理。训练与预测使用同一种数据组织（整段序列 + 逐帧标签）；正式评测由 benchmark 调用本
Pipeline 的 ``predict()`` 后完成。

模型作为可替换组件（GRU / MS-TCN / MS-TCN++…）由 ``model.type`` 选取，只需满足
``[B,T,F] -> [B,T,C]`` 的前向约定；监督口径（逐帧 CE、类别加权）与推理方式（逐帧 argmax）
由本流水线拥有，不写在模型里。数据读取与训练验证摘要由两条时序流水线共享，但绝不跨到
detection 域。

两个 **可选 duck-type 钩子**（有则调、无则退化，不写基类）让个别模型携带自身的训练细节而
不污染通用脊柱：``fit_normalization(features)`` 训练前按训练集统计写归一化 buffer；
``compute_loss(x, y, criterion)`` 让模型自持训练配方（如 MS-TCN++ 的多 stage 深监督 +
T-MSE），流水线仍把类别加权 CE 作为监督口径传入。缺钩子的模型走默认单前向逐帧 CE。
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # tqdm 可选，缺失时退化为原样迭代
    def tqdm(iterable, **_kwargs):
        return iterable

from ..core.checkpoint import load_checkpoint, load_training_checkpoint, save_training_checkpoint
from ..core.environment import now_stamp, set_seed
from ..core.execution import PredictionOutput, format_params
from ..core.history import HistoryWriter, try_plot_training_history
from ..core.integrity import check_feature_schema
from ..core.pipeline import Pipeline
from ..core.run import RunContext
from .data import (
    apply_target_mask_augmentation,
    assert_resume_dataset_compatible,
    build_dataset_provenance,
    build_temporal_meta,
    load_split,
    resolve_mask_target_ids,
    resolve_external_temporal_meta,
    resolve_target_mask_augmentation,
    split_video_names,
)
from .external import configure_external_model
from .models import build_model
from .training_validation import summarize_training_metrics
from .util import compute_class_weights


def _load_eval_model(cfg: dict, ckpt: str, device):
    """优先按 sidecar 重建模型；探索模式可显式改用 YAML 重建外部裸权重。"""
    model_cfg = cfg["model"]
    expected = {"type": model_cfg["type"], "input_dim": model_cfg["input_dim"], "num_classes": model_cfg["num_classes"]}
    mode = (cfg.get("evaluation") or {}).get("mode", "formal")
    state_dict, meta = load_checkpoint(
        ckpt,
        expected=expected,
        map_location=device,
        require_meta_schema=mode == "formal",
        fallback_meta=resolve_external_temporal_meta(cfg, "full_sequence_temporal"),
    )
    model = build_model(meta["model"]).to(device)
    model.load_state_dict(state_dict, strict=True)
    configure_external_model(model, cfg, meta)
    if meta.get("num_params") is None:
        meta["num_params"] = sum(parameter.numel() for parameter in model.parameters())
    model.eval()
    return model, meta


def _infer_split(model, features, device) -> list:
    """逐视频整段前向 + 逐帧 argmax，返回与 features 对齐的预测序列列表。"""
    preds = []
    with torch.no_grad():
        for feats in features:
            x = torch.from_numpy(feats).float().unsqueeze(0).to(device)  # [1, T, F]
            logits = model(x)  # [1, T, C]
            preds.append(torch.argmax(logits[0], dim=-1).cpu().numpy().astype(np.int64))
    return preds


def _loss_is_finite(loss: torch.Tensor) -> bool:
    """训练可靠性护栏：NaN/Inf loss 立即中断并写入 run status。"""

    return bool(torch.isfinite(loss.detach()).cpu().item())


def _validation_split_name(cfg: dict) -> str:
    """训练期 validation 优先用 split_val；旧配置没有时回退 split_eval。"""

    data = cfg["data"]
    return data.get("split_val") or data["split_eval"]


def _evaluate_full_sequence(model, features, truths, id2name, criterion, device) -> dict:
    """按视频整段 validation，返回 loss 与指标，保持视频边界不用于训练更新。"""

    model.eval()
    losses: list[float] = []
    preds = []
    with torch.no_grad():
        for feats, truth in zip(features, truths):
            x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
            y = torch.tensor(truth, dtype=torch.long).unsqueeze(0).to(device)
            if hasattr(model, "compute_loss"):
                loss = model.compute_loss(x, y, criterion)
                logits = model(x)
            else:
                logits = model(x)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            losses.append(float(loss.detach().cpu().item()))
            preds.append(torch.argmax(logits[0], dim=-1).cpu().numpy().astype(np.int64))
    pred_by_item = {
        f"video-{index:04d}": [id2name[int(value)] for value in video]
        for index, video in enumerate(preds)
    }
    truth_by_item = {
        f"video-{index:04d}": [id2name[int(value)] for value in video]
        for index, video in enumerate(truths)
    }
    return {
        "val_loss": float(np.mean(losses)) if losses else None,
        **summarize_training_metrics(
            pred_by_item,
            truth_by_item,
            list(id2name.values()),
        ),
    }


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


class FullSequenceTemporalPipeline(Pipeline):
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
        resolve_mask_target_ids(data, cfg.get("feature_schema"))
        resolve_target_mask_augmentation(data, cfg.get("augmentation"))

    def train(self, cfg: dict, runs_dir: str, seed: int, device) -> str:
        train_cfg = cfg["train"]
        model_cfg = cfg["model"]

        run = RunContext(runs_dir, label=model_cfg["type"])
        run.save_config(cfg)
        run.save_env(device, seed=seed)
        run.write_status("running", stage="initializing")

        current_epoch = None
        try:
            set_seed(seed)
            model = build_model(model_cfg).to(device)
            dataset_provenance = build_dataset_provenance(
                cfg["data"], cfg.get("feature_schema")
            )

            # data（features 契约：读 ActionMixed + bbox→40维）+ 训练前 schema 兼容检查。
            features, truths, _ = load_split(
                cfg["data"],
                cfg["data"]["split_train"],
                feature_schema=cfg.get("feature_schema"),
            )
            features = apply_target_mask_augmentation(
                features,
                cfg["data"],
                cfg.get("augmentation"),
                seed=seed,
            )
            problems = check_feature_schema(features[0].shape[1], cfg.get("feature_schema"))
            if problems:
                raise ValueError("特征 schema 与配置不兼容:\n  - " + "\n  - ".join(problems))
            val_split = _validation_split_name(cfg)
            val_features, val_truths, val_id2name = load_split(
                cfg["data"], val_split, feature_schema=cfg.get("feature_schema")
            )

            resume_path = train_cfg.get("resume")
            # 可选归一化钩子：resume 时状态会从 checkpoint 恢复，避免重新 fit 改变统计。
            if hasattr(model, "fit_normalization") and not resume_path:
                model.fit_normalization(features)

            # 整段 + 逐帧：变长全序列逐条喂入（batch_size=1，不做批内对齐）。
            train_ds = ConcatDataset(
                [FullSequenceDataset(features[i], truths[i]) for i in range(len(features))]
            )
            train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

            weights = compute_class_weights(train_loader, num_classes=model_cfg["num_classes"])
            criterion = nn.CrossEntropyLoss(
                weight=torch.tensor([weights[i] for i in sorted(weights)], dtype=torch.float32).to(device)
            )
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_cfg.get("lr", 1e-3),
                weight_decay=train_cfg.get("weight_decay", 0.0),
            )
            grad_clip = train_cfg.get("grad_clip")  # 值驱动：缺省则不裁剪
            start_epoch = 1
            best_metric = {"name": "val_acc", "mode": "max", "value": None, "epoch": None}

            if resume_path:
                expected = {"type": model_cfg["type"], "input_dim": model_cfg["input_dim"], "num_classes": model_cfg["num_classes"]}
                payload, _meta = load_training_checkpoint(resume_path, expected=expected, map_location=device)
                assert_resume_dataset_compatible(_meta, dataset_provenance)
                model.load_state_dict(payload["model_state"])
                optimizer.load_state_dict(payload["optimizer_state"])
                start_epoch = int(payload["epoch"]) + 1
                best_metric.update(payload.get("best_metric") or {})
                run.write_status("running", stage="resumed", resume=str(resume_path), start_epoch=start_epoch)

            extra = {"normalizer": "zscore/train-set/buffers/v1"} if hasattr(model, "fit_normalization") else None
            meta = build_temporal_meta(
                model_cfg,
                cfg.get("feature_schema", {}),
                pipeline=self.pipeline_name,
                window=None,  # 全序列不加窗
                num_params=sum(p.numel() for p in model.parameters()),
                train_cfg=train_cfg,
                trained_at=now_stamp(),
                augmentation=cfg.get("augmentation"),
                dataset=dataset_provenance,
                extra=extra,
            )
            history = HistoryWriter(
                run.history_path,
                ["epoch", "train_loss", "val_loss", "val_acc", "val_edit", "val_f1_0.5", "lr", "epoch_sec", "checkpoint_best", "checkpoint_last", "status"],
            )
            best_path = run.checkpoints_dir / "best.pt"
            last_path = run.checkpoints_dir / "last.pt"

            epochs = train_cfg.get("epochs", 20)
            if start_epoch > epochs:
                run.write_status(
                    "succeeded",
                    stage="already_complete",
                    best_metric=best_metric,
                    best_checkpoint=str(best_path),
                    last_checkpoint=str(last_path),
                )
                return str(best_path if best_path.exists() else last_path)

            for epoch in tqdm(range(start_epoch, epochs + 1), desc="train"):
                current_epoch = epoch
                epoch_start = time.perf_counter()
                model.train()
                losses: list[float] = []
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    if hasattr(model, "compute_loss"):
                        loss = model.compute_loss(x, y, criterion)
                    else:
                        logits = model(x)  # [B, T, C]
                        loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                    if not _loss_is_finite(loss):
                        raise FloatingPointError(f"epoch={epoch}: loss is NaN/Inf")
                    optimizer.zero_grad()
                    loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu().item()))

                validation = _evaluate_full_sequence(model, val_features, val_truths, val_id2name, criterion, device)
                train_loss = float(np.mean(losses)) if losses else None
                val_acc = validation["val_acc"]
                improved = val_acc is not None and (
                    best_metric["value"] is None or val_acc > float(best_metric["value"])
                )
                if improved:
                    best_metric.update({"value": val_acc, "epoch": epoch})
                    save_training_checkpoint(best_path, model=model, optimizer=optimizer, epoch=epoch, meta=meta, best_metric=best_metric)
                save_training_checkpoint(last_path, model=model, optimizer=optimizer, epoch=epoch, meta=meta, best_metric=best_metric)
                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **validation,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_sec": round(time.perf_counter() - epoch_start, 4),
                    "checkpoint_best": str(best_path) if best_path.exists() else "",
                    "checkpoint_last": str(last_path),
                    "status": "ok",
                }
                history.append(row)
                run.write_status("running", stage="epoch_complete", epoch=epoch, best_metric=best_metric, last_checkpoint=str(last_path))

            curves_path, curves_error = try_plot_training_history(
                run.history_path,
                run.dir / "training_curves.png",
            )
            run.write_status(
                "succeeded",
                best_metric=best_metric,
                best_checkpoint=str(best_path),
                last_checkpoint=str(last_path),
                history=str(run.history_path),
                training_curves=str(curves_path) if curves_path else None,
                training_curves_error=curves_error,
            )
            print(f"[train] run_dir={run.dir}")
            print(f"[train] best_checkpoint={best_path}")
            print(f"[train] last_checkpoint={last_path}")
            if curves_path:
                print(f"[train] training_curves={curves_path}")
            elif curves_error:
                print(f"[train] training_curves skipped: {curves_error}")
            return str(best_path if best_path.exists() else last_path)
        except Exception as exc:
            run.write_exception_status(exc, epoch=current_epoch)
            raise

    def predict(self, cfg: dict, ckpt: str, device) -> PredictionOutput:
        """运行全序列模型，返回不含指标判分的逐视频预测事实。"""

        model, meta = _load_eval_model(cfg, ckpt, device)
        limits = (cfg.get("evaluation") or {}).get("limits") or {}
        features, truths, id2name = load_split(
            cfg["data"],
            cfg["data"]["split_eval"],
            feature_schema=cfg.get("feature_schema"),
            max_videos=limits.get("max_videos"),
            max_frames=limits.get("max_frames"),
        )

        video_preds = _infer_split(model, features, device)
        names = split_video_names(
            cfg["data"],
            cfg["data"]["split_eval"],
            max_videos=limits.get("max_videos"),
            max_frames=limits.get("max_frames"),
        )
        pred_by_item = {
            name: [id2name[int(value)] for value in video]
            for name, video in zip(names, video_preds)
        }
        truth_by_item = {
            name: [id2name[int(value)] for value in video]
            for name, video in zip(names, truths)
        }
        labels = list(id2name.values())

        return PredictionOutput(
            model_type=meta["type"],
            model_id=f"{meta['type']}-{format_params(meta.get('num_params'))}",
            pipeline=self.pipeline_name,
            checkpoint=str(ckpt),
            dataset=cfg["data"].get("name", cfg["data"].get("root")),
            predictions=pred_by_item,
            targets=truth_by_item,
            labels=labels,
            # 实际输入变换来自本次评估配置；mask_targets 可能用于已有 checkpoint 的遮罩实验。
            feature_schema=cfg.get("feature_schema", meta.get("feature_schema", {})),
            inference_semantics=dict(FULL_SEQUENCE_SEMANTICS),
            num_params=meta.get("num_params"),
            metadata={
                "split": cfg["data"]["split_eval"],
                "input_dim": cfg["model"]["input_dim"],
                "input_shape": [1, "T", cfg["model"]["input_dim"]],
                "checkpoint_format": meta.get("_checkpoint_format", "state_dict"),
                "checkpoint_metadata_source": meta.get("source", "sidecar"),
                "checkpoint_metadata_bound": bool(
                    (meta.get("_metadata_integrity") or {}).get("bound")
                ),
            },
        )
