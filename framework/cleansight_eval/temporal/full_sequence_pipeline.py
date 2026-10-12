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

import math
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
from ..core.history import HistoryWriter, temporal_history_columns, try_plot_training_history
from ..core.integrity import check_feature_schema
from ..core.pipeline import Pipeline
from ..core.run import RunContext
from .data import (
    apply_sequence_augmentation,
    apply_target_mask_augmentation,
    assert_resume_dataset_compatible,
    build_dataset_provenance,
    build_temporal_meta,
    load_split,
    resolve_image_feature_dim,
    resolve_mask_target_ids,
    resolve_external_temporal_meta,
    resolve_sequence_augmentation,
    resolve_target_mask_augmentation,
    resolve_train_video_fraction,
    resolve_train_video_limit,
    split_video_names,
)
from .external import configure_external_model
from .models import build_model
from .training_validation import summarize_training_metrics
from .util import (
    TransitionWeightedCE,
    compute_class_weights,
    estimate_transition_rarity,
    resolve_best_metric,
    resolve_class_weight_clip,
)


VALID_LR_SCHEDULES = ("constant", "cosine")


def resolve_lr_schedule(train_cfg: dict) -> tuple[str, int, float]:
    """解析学习率调度配置 → ``(schedule, warmup_epochs, min_lr_ratio)``。

    - ``train.lr_schedule``：``constant``（默认，等价于历史行为，**不改既有口径**）或 ``cosine``；
    - ``train.warmup_epochs``：前若干轮从 ``lr/warmup`` 线性升到 ``lr``（0 = 不 warmup）；
    - ``train.min_lr_ratio``：余弦衰减的下界比例（0 = 衰减到 0）。

    非法值立即报错，不静默退回默认——避免"配了调度却没生效"的静默对照污染。
    """

    raw = str((train_cfg or {}).get("lr_schedule") or "constant").lower()
    if raw not in VALID_LR_SCHEDULES:
        raise ValueError(f"train.lr_schedule 必须是 {sorted(VALID_LR_SCHEDULES)} 之一，实际 {raw!r}")
    try:
        warmup = int((train_cfg or {}).get("warmup_epochs") or 0)
        min_ratio = float((train_cfg or {}).get("min_lr_ratio") or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError("train.warmup_epochs / train.min_lr_ratio 必须是数字") from exc
    if warmup < 0:
        raise ValueError(f"train.warmup_epochs 不能为负，实际 {warmup}")
    if not (0.0 <= min_ratio <= 1.0):
        raise ValueError(f"train.min_lr_ratio 必须在 [0, 1]，实际 {min_ratio}")
    return raw, warmup, min_ratio


def lr_factor(schedule: str, epoch: int, epochs: int, warmup_epochs: int, min_lr_ratio: float) -> float:
    """第 ``epoch`` 轮（1-based）的学习率**倍率**。

    ``constant`` 恒为 1.0；``cosine`` 先线性 warmup（若配置），再按余弦从 1 衰减到
    ``min_lr_ratio``。写成显式函数（而非 ``LambdaLR`` 的状态机）是为了**与 resume 天然兼容**：
    倍率只依赖轮号，不依赖调度器内部状态。

    **衰减区间口径（固定，不随实现漂移）**：余弦跨度为 ``(warmup_epochs, epochs]``，
    即 ``progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)``——
    第 ``warmup_epochs`` 轮为峰值 1.0、末轮恰为 ``min_lr_ratio``。
    因此 ``warmup_epochs=0`` 时首轮已是 ``cos(π/epochs)`` 而非 1.0（``epochs=10`` 时 ≈0.976）。
    """

    if schedule == "constant":
        return 1.0
    if warmup_epochs and epoch <= warmup_epochs:
        return max(1e-8, epoch / float(warmup_epochs))
    span = max(1, epochs - warmup_epochs)
    progress = min(max((epoch - warmup_epochs) / float(span), 0.0), 1.0)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


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
        resolve_image_feature_dim(model, cfg.get("feature_schema"))
        resolve_target_mask_augmentation(data, cfg.get("augmentation"))
        resolve_sequence_augmentation(data, cfg.get("augmentation"))  # 早校验：非法 sigma/max_rate
        resolve_train_video_fraction(data)  # 校验 (0, 1] 范围，非法值直接报错
        resolve_class_weight_clip((cfg.get("train") or {}).get("class_weight_clip"))  # 早校验
        train = cfg.get("train", {})
        # 转移级代价敏感的权重必须是非负有限数（0 = 关闭），非法值在训练前直接报错。
        raw_transition = train.get("transition_loss_weight")
        if raw_transition is not None:
            try:
                value = float(raw_transition)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"train.transition_loss_weight 必须是数字（0 = 关闭），实际 {raw_transition!r}"
                ) from exc
            if not (value >= 0 and value == value and value != float("inf")):
                raise ValueError(
                    f"train.transition_loss_weight 必须 ≥0 且有限，实际 {raw_transition!r}"
                )
        resolve_best_metric(train)  # 只做校验：非法选点口径在训练前直接报错
        patience = train.get("patience")
        if patience is not None and (not isinstance(patience, int) or patience < 1):
            raise ValueError("train.patience 必须是 ≥1 的整数（按 val_loss 早停，缺省关闭）")

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
                # 训练子采样（data.train_video_fraction）：只作用于 train split，val/test 全量评估。
                video_limit=resolve_train_video_limit(cfg["data"], cfg["data"]["split_train"]),
            )
            features = apply_target_mask_augmentation(
                features,
                cfg["data"],
                cfg.get("augmentation"),
                seed=seed,
                feature_schema=cfg.get("feature_schema"),
            )
            # 序列级增强（feature_jitter / temporal_scale）在目标遮罩之后：前者按「非零元素」
            # 判定有无检测，遮罩产生的零不参与抖动。
            features, truths = apply_sequence_augmentation(
                features,
                truths,
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

            weights = compute_class_weights(
                train_loader, num_classes=model_cfg["num_classes"],
                clip=train_cfg.get("class_weight_clip"),
            )
            class_weight_tensor = torch.tensor(
                [weights[i] for i in sorted(weights)], dtype=torch.float32
            ).to(device)
            # 转移级代价敏感（可选）：逐帧权重只依赖真值，封装成 criterion 即可让模型自持配方
            # （如 mstcn2 的多 stage 深监督 + T-MSE）一并带上，无需改动任何模型文件。
            transition_strength = float(train_cfg.get("transition_loss_weight") or 0.0)
            if transition_strength > 0:
                criterion = TransitionWeightedCE(
                    class_weight_tensor,
                    estimate_transition_rarity(truths, model_cfg["num_classes"]),
                    transition_strength,
                    single_sequence=True,  # 全序列流水线固定 batch_size=1（见上方 DataLoader）
                )
                run.write_status(
                    "running", stage="criterion",
                    transition_loss_weight=transition_strength,
                )
            else:
                # 标签平滑（train.label_smoothing，默认 0 = 与历史行为逐位一致）：
                # LS timeline 标注本身有噪声，平滑可抑制过度自信、改善泛化。
                criterion = nn.CrossEntropyLoss(
                    weight=class_weight_tensor,
                    label_smoothing=float(train_cfg.get("label_smoothing") or 0.0),
                )
            base_lr = float(train_cfg.get("lr", 1e-3))
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=base_lr,
                weight_decay=train_cfg.get("weight_decay", 0.0),
            )
            # 学习率调度：默认 constant（与历史行为逐位一致）；cosine 为可选改进。
            lr_schedule, warmup_epochs, min_lr_ratio = resolve_lr_schedule(train_cfg)
            label_smoothing = float(train_cfg.get("label_smoothing") or 0.0)
            if not (0.0 <= label_smoothing < 1.0):
                raise ValueError(f"train.label_smoothing 必须在 [0, 1)，实际 {label_smoothing}")
            if label_smoothing and transition_strength:
                # 两条路径互斥：TransitionWeightedCE 不接受 label_smoothing，
                # 同时开会让平滑被静默忽略——按仓库纪律直接报错而不是"配了没生效"。
                raise ValueError(
                    "train.label_smoothing 与 train.transition_loss_weight 不能同时开启"
                    "（transition 路径的 criterion 不支持标签平滑）"
                )
            grad_clip = train_cfg.get("grad_clip")  # 值驱动：缺省则不裁剪
            start_epoch = 1
            best_metric = {"name": resolve_best_metric(train_cfg), "mode": "max", "value": None, "epoch": None}

            if resume_path:
                expected = {"type": model_cfg["type"], "input_dim": model_cfg["input_dim"], "num_classes": model_cfg["num_classes"]}
                payload, _meta = load_training_checkpoint(resume_path, expected=expected, map_location=device)
                assert_resume_dataset_compatible(_meta, dataset_provenance)
                model.load_state_dict(payload["model_state"])
                optimizer.load_state_dict(payload["optimizer_state"])
                start_epoch = int(payload["epoch"]) + 1
                best_metric.update(payload.get("best_metric") or {})
                run.write_status("running", stage="resumed", resume=str(resume_path), start_epoch=start_epoch)

            # 归一化口径溯源：模型自己声明（GRU 默认直通返回 None；MS-TCN 恒为 z-score）。
            # 无 normalizer_spec 的历史模型（外部 checkpoint）保持原样声明。
            spec_fn = getattr(model, "normalizer_spec", None)
            if callable(spec_fn):
                spec = spec_fn()
            elif hasattr(model, "fit_normalization"):
                spec = "zscore/train-set/buffers/v1"
            else:
                spec = None
            extra = {"normalizer": spec} if spec else None
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
                temporal_history_columns(),
            )
            best_path = run.checkpoints_dir / "best.pt"
            last_path = run.checkpoints_dir / "last.pt"

            epochs = train_cfg.get("epochs", 20)
            # best checkpoint 指标与早停（与滑窗流水线同口径，2026-09 配方修复）
            best_metric_name = resolve_best_metric(train_cfg)
            patience = train_cfg.get("patience")
            no_improve_epochs = 0
            best_val_loss = float("inf")
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
                # 逐轮设置学习率：倍率只依赖轮号，故 resume 后仍落在同一条曲线上。
                if lr_schedule != "constant":
                    scheduled = base_lr * lr_factor(lr_schedule, epoch, epochs, warmup_epochs, min_lr_ratio)
                    for group in optimizer.param_groups:
                        group["lr"] = scheduled
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
                metric_value = validation.get(best_metric_name)
                improved = metric_value is not None and (
                    best_metric["value"] is None or metric_value > float(best_metric["value"])
                )
                if improved:
                    best_metric.update({"value": metric_value, "epoch": epoch})
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

                if patience:
                    val_loss = validation.get("val_loss")
                    if val_loss is not None and val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        print(f"[train] early stop: val_loss 连续 {patience} epoch 未改善（best {best_val_loss:.4f}），停在 epoch {epoch}")
                        break

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
