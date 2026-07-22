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
from ..core.execution import PredictionOutput, format_params, sample_callable_latency
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
from .metrics import compute_temporal_metrics_by_item
from .models import build_model, is_causal
from .util import causal_decision, compute_class_weights

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


def _loss_is_finite(loss: torch.Tensor) -> bool:
    """训练可靠性护栏：NaN/Inf loss 立即中断并写入 run status。"""

    return bool(torch.isfinite(loss.detach()).cpu().item())


def _metric_value(metrics: dict, name: str):
    value = metrics.get(name)
    return None if value is None or value.value is None else float(value.value)


def _validation_split_name(cfg: dict) -> str:
    """训练期 validation 优先用 split_val；旧配置没有时回退 split_eval。"""

    data = cfg["data"]
    return data.get("split_val") or data["split_eval"]


def _evaluate_sliding_window(model, datasets, id2name, criterion, device) -> dict:
    """按视频逐窗 validation，返回 loss 与末帧预测指标。"""

    model.eval()
    losses: list[float] = []
    video_preds, video_gts = [], []
    with torch.no_grad():
        for ds in datasets:
            preds = []
            gts = []
            for i in range(len(ds)):
                x, y = ds[i]
                x = x.unsqueeze(0).to(device)
                y = y.unsqueeze(0).to(device)
                logits = model(x)
                loss = criterion(logits[:, -1, :], y)
                losses.append(float(loss.detach().cpu().item()))
                preds.append(int(torch.argmax(logits[0, -1], dim=-1).cpu().item()))
                gts.append(int(y.cpu().item()))
            video_preds.append(np.asarray(preds, dtype=np.int64))
            video_gts.append(np.asarray(gts, dtype=np.int64))

    pred_by_item = {
        f"video-{index:04d}": [id2name[int(value)] for value in video]
        for index, video in enumerate(video_preds)
    }
    truth_by_item = {
        f"video-{index:04d}": [id2name[int(value)] for value in video]
        for index, video in enumerate(video_gts)
    }
    metrics = compute_temporal_metrics_by_item(pred_by_item, truth_by_item, list(id2name.values()))
    return {
        "val_loss": float(np.mean(losses)) if losses else None,
        "val_acc": _metric_value(metrics, "acc"),
        "val_edit": _metric_value(metrics, "edit"),
        "val_f1_0.5": _metric_value(metrics, "f1@0.5"),
    }


class SlidingWindowTemporalPipeline(Pipeline):
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
        resolve_mask_target_ids(data, cfg.get("feature_schema"))
        resolve_target_mask_augmentation(data, cfg.get("augmentation"))

    def train(self, cfg: dict, runs_dir: str, seed: int, device) -> str:
        train_cfg = cfg["train"]
        model_cfg = cfg["model"]
        window = train_cfg.get("window", 64)

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

            features, truths, _ = load_split(
                cfg["data"],
                cfg["data"]["split_train"],
                window=window,
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
                cfg["data"],
                val_split,
                window=window,
                feature_schema=cfg.get("feature_schema"),
            )

            resume_path = train_cfg.get("resume")
            if hasattr(model, "fit_normalization") and not resume_path:
                model.fit_normalization(features)

            # 窗口 + 末帧标签：窗口样本可批处理，用 cfg 的 batch_size。
            train_ds = ConcatDataset(
                [SlidingWindowDataset(features[i], truths[i], window) for i in range(len(features))]
            )
            train_loader = DataLoader(train_ds, batch_size=train_cfg.get("batch_size", 32), shuffle=True)
            val_datasets = [SlidingWindowDataset(val_features[i], val_truths[i], window) for i in range(len(val_features))]

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
                window=window,
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
                    logits = model(x)  # [B, window, C]
                    loss = criterion(logits[:, -1, :], y)
                    if not _loss_is_finite(loss):
                        raise FloatingPointError(f"epoch={epoch}: loss is NaN/Inf")
                    optimizer.zero_grad()
                    loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu().item()))

                validation = _evaluate_sliding_window(model, val_datasets, val_id2name, criterion, device)
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
        """运行因果滑窗模型，返回不含指标判分的逐视频预测事实。"""

        model_cfg = cfg["model"]
        expected = {"type": model_cfg["type"], "input_dim": model_cfg["input_dim"], "num_classes": model_cfg["num_classes"]}
        mode = (cfg.get("evaluation") or {}).get("mode", "formal")
        state_dict, meta = load_checkpoint(
            ckpt,
            expected=expected,
            map_location=device,
            require_meta_schema=mode == "formal",
            fallback_meta=resolve_external_temporal_meta(cfg, self.pipeline_name),
        )

        model = build_model(meta["model"]).to(device)
        model.load_state_dict(state_dict, strict=True)
        if meta.get("num_params") is None:
            meta["num_params"] = sum(parameter.numel() for parameter in model.parameters())

        window = meta.get("window") or cfg["train"].get("window", 64)
        features, truths, id2name = load_split(
            cfg["data"],
            cfg["data"]["split_eval"],
            window=window,
            feature_schema=cfg.get("feature_schema"),
        )
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

        names = split_video_names(cfg["data"], cfg["data"]["split_eval"], window=window)
        pred_by_item = {
            name: [id2name[int(value)] for value in video]
            for name, video in zip(names, video_preds)
        }
        truth_by_item = {
            name: [id2name[int(value)] for value in video]
            for name, video in zip(names, video_gts)
        }
        labels = list(id2name.values())
        semantics = {
            "mode": "windowed_causal",
            "sees": "causal_sliding_window",
            "window": window,
            "advance": 1,
            "cold_start": f"前 {window - 1} 帧填充 idle",
            "reset": "per_video",
            "smoothing": f"causal_decision(min_duration={MIN_DURATION})",
        }

        timing = {}
        evaluation_cfg = cfg.get("evaluation", {})
        if evaluation_cfg.get("measure_latency", True):
            latency_input = torch.randn(1, window, model_cfg["input_dim"], device=device)

            def _tick():
                return model(latency_input)[0, -1]

            timing = sample_callable_latency(
                _tick,
                device,
                warmup=int(evaluation_cfg.get("latency_warmup", 20)),
                runs=int(evaluation_cfg.get("latency_runs", 200)),
                scope="model_forward_single_window",
                context={
                    "window": window,
                    "input_dim": model_cfg["input_dim"],
                    "input_shape": [1, window, model_cfg["input_dim"]],
                    "output": "last_frame_logits",
                },
            )

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
            inference_semantics=semantics,
            num_params=meta.get("num_params"),
            timing=timing,
            metadata={
                "split": cfg["data"]["split_eval"],
                "window": window,
                "input_dim": model_cfg["input_dim"],
                "input_shape": [1, window, model_cfg["input_dim"]],
                "checkpoint_metadata_source": meta.get("source", "sidecar"),
                "checkpoint_metadata_bound": bool(
                    (meta.get("_metadata_integrity") or {}).get("bound")
                ),
            },
        )
