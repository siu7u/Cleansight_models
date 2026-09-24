"""ROI 图像分类流水线（roi_classification）。

特征融合模型的标准 Pipeline 实现：训练消费 YOLO 分组数据集的 GT 框裁剪 ROI 数据，
训练 CNN backbone + MLP 多标签头；预测对 ROI 图像块逐类输出 sigmoid 概率。
checkpoint 复用 ``core.checkpoint.save_checkpoint`` 的绑定 meta 契约。

torch/torchvision 为重依赖，全部在方法内部 import（与 detection/yolo.py 惯例一致），
使仅做配置/边界校验的场景无需安装它们。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..core.checkpoint import load_checkpoint, save_checkpoint
from ..core.environment import now_stamp, set_seed
from ..core.execution import PredictionOutput
from ..core.metrics import (
    classification_training_keys,
    classification_training_values,
    confusion_counts,
    decide,
    merge_counts,
    metrics_from_counts,
    multilabel_metrics,
)
from ..core.pipeline import Pipeline
from ..core.run import RunContext
from .data import build_roi_dataset, load_dataset, save_dataset
from .model import BACKBONE_CONFIGS, FeatureFusionModel


# 训练期可选点指标：val_loss（历史默认，越小越好）+ 注册表里的全部分类指标（越大越好）。
# 与两条时序流水线的 train.best_metric 同一机制；时序侧只允许指标（不含 val_loss），
# 分类侧保留 val_loss 以兼容既有配方。
CLASSIFICATION_BEST_METRICS: tuple[str, ...] = ("val_loss", *classification_training_keys())


def best_metric_mode(name: str) -> str:
    """选点方向：``val_loss`` 越小越好，其余（precision/recall/f1/exact_match）越大越好。"""

    return "min" if name == "val_loss" else "max"


def metric_improved(value: float | None, best: float | None, mode: str) -> bool:
    """按方向判断当前 epoch 是否优于历史最优；``None``（样本不足）永不视为改善。"""

    if value is None:
        return False
    if best is None:
        return True
    return value < best if mode == "min" else value > best


class _RoiTensorDataset:
    """按需把 uint8 BGR 的 ROI 转成 float32 RGB 张量（内存峰值 ≈ 一个 batch）。

    历史实现是 `torch.from_numpy(X).float()` 一次性把整个数据集转成 float32（2 万张 224×224
    的裁剪 = 约 12 GB），在 WSL 这类内存受限环境里会被 OOM kill。这里改成逐样本转换：
    只在 ``__getitem__`` 里复制单张图（约 600 KB），原始数组可以是 ``np.load(..., mmap_mode="r")``
    的内存映射，不常驻内存。
    """

    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        import numpy as np
        import torch

        row = int(self.indices[position])
        # np.array(..., copy=True) 而非 ascontiguousarray：内存映射的整行本身就是连续视图，
        # ascontiguousarray 会原样返回只读 memmap 视图，torch.from_numpy 会警告"非可写数组"。
        image = np.array(self.images[row], copy=True)           # 只复制这一张（约 600 KB）
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC uint8 -> CHW float32
        return tensor[[2, 1, 0]], torch.from_numpy(np.asarray(self.labels[row])).float()  # BGR->RGB


class ClassificationPipeline(Pipeline):
    pipeline_name = "roi_classification"

    def validate_config(self, cfg: dict) -> None:
        model = cfg.get("model", {})
        if model.get("type") != "feature_fusion":
            raise ValueError("分类流水线 model.type 需为 feature_fusion")
        backbone = model.get("backbone", "resnet50")
        if backbone not in BACKBONE_CONFIGS:
            raise ValueError(f"不支持的 backbone: {backbone}")
        data = cfg.get("data", {})
        if not data.get("classes"):
            raise ValueError("分类流水线 data 段需包含 classes（目标类别名列表）")
        if not data.get("group_dir"):
            raise ValueError("分类流水线 data 段需包含 group_dir（YOLO 分组数据集目录）")
        best_metric = (cfg.get("train") or {}).get("best_metric")
        if best_metric is not None and best_metric not in CLASSIFICATION_BEST_METRICS:
            raise ValueError(
                f"train.best_metric 必须是 {list(CLASSIFICATION_BEST_METRICS)} 之一，实际 {best_metric!r}"
            )

    def _dataset_dir(self, data_cfg: dict) -> Path:
        from ..core.run import RunContext  # noqa: F401  (保持导入一致性)

        base = Path(data_cfg.get("dataset_dir") or "runs/feature_fusion/datasets")
        if not base.is_absolute():
            base = Path(__file__).resolve().parents[3] / base
        return base

    def train(self, cfg: dict, runs_dir: str, seed: int, device) -> str:
        set_seed(seed)
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        train_cfg = cfg.get("train", {})
        classes = list(data_cfg["classes"])
        group_dir = Path(data_cfg["group_dir"])
        if not group_dir.is_absolute():
            group_dir = Path(__file__).resolve().parents[3] / group_dir

        run = RunContext(runs_dir, label="classification")
        run.save_config(cfg)
        run.save_env(device, seed=seed)
        run.write_status("running", stage="initializing")

        try:
            # 构建或加载 ROI 数据集
            ds_base = self._dataset_dir(data_cfg)
            cache_key = "-".join(classes)
            cache_dir = ds_base / cache_key
            try:
                X, y, actual_classes = load_dataset(classes, ds_base)
            except FileNotFoundError:
                X, y, actual_classes, stats = build_roi_dataset(
                    group_dir,
                    classes,
                    roi_size=int(model_cfg.get("roi_size", 224)),
                    neg_ratio=float(data_cfg.get("neg_ratio", 1.0)),
                )
                save_dataset(X, y, actual_classes, stats, ds_base)
            classes = actual_classes

            model = FeatureFusionModel(
                num_classes=len(classes),
                backbone_name=model_cfg.get("backbone", "resnet50"),
                freeze_backbone=bool(model_cfg.get("freeze_backbone", False)),
                hidden_dim=int(model_cfg.get("hidden_dim", 256)),
                dropout=float(model_cfg.get("dropout", 0.3)),
            )
            model.to_device(str(device))

            best_metric = str(train_cfg.get("best_metric") or "val_loss")
            history, best_state, best = self._fit(
                model, X, y, classes,
                epochs=int(train_cfg.get("epochs", 50)),
                batch_size=int(train_cfg.get("batch_size", 32)),
                lr=float(train_cfg.get("lr", 1e-3)),
                val_split=float(data_cfg.get("val_split", 0.2)),
                best_metric=best_metric,
            )

            # 保存 checkpoint + 绑定 meta
            model.load_state_dict(best_state)
            ckpt = run.checkpoints_dir / "best.pt"
            meta = {
                "type": "feature_fusion",
                "pipeline": self.pipeline_name,
                "num_classes": len(classes),
                "classes": classes,
                "backbone": model_cfg.get("backbone", "resnet50"),
                "input_size": BACKBONE_CONFIGS[model_cfg.get("backbone", "resnet50")]["input_size"],
                "model": model_cfg,
                "train": train_cfg,
                "data": data_cfg,
                "trained_at": now_stamp(),
                # best_val_loss 保留为历史字段；best_metric 结构与时序侧一致（name/value/epoch/mode）。
                "best_val_loss": best["val_loss"],
                "best_metric": best,
            }
            save_checkpoint(ckpt, model.state_dict(), meta)

            history_path = run.dir / "history.json"
            history_path.write_text(json.dumps(history, indent=2, default=str),
                                    encoding="utf-8")
            run.write_status("succeeded", best_checkpoint=str(ckpt),
                             best_metric=best, best_val_loss=best["val_loss"],
                             history=str(history_path))
            print(f"[train] run_dir={run.dir}")
            print(f"[train] checkpoint={ckpt}")
            print(f"[train] best_metric={best['name']} value={best['value']} epoch={best['epoch']}")
            return str(ckpt)
        except Exception as exc:
            run.write_exception_status(exc)
            raise

    def _fit(self, model, X, y, classes, *, epochs, batch_size, lr, val_split,
             best_metric: str = "val_loss"):
        """训练循环，返回 ``(history, best_state_dict, best_metric_info)``。

        ``best_metric`` 取自 :data:`CLASSIFICATION_BEST_METRICS`（默认 ``val_loss``，与历史行为一致）；
        每个 epoch 的验证指标按注册表口径计算，选点方向由 :func:`best_metric_mode` 给出。
        """

        import numpy as np
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader

        # 只切**索引**：sklearn 的划分只依赖 (n, random_state, stratify 标签)，对索引数组切分
        # 得到的 train/val 划分与直接切 X 完全一致，但不会复制两份 uint8 大数组。
        index = np.arange(len(y))
        train_idx, val_idx = train_test_split(
            index, test_size=val_split, random_state=42,
            stratify=y.any(axis=1) if y.shape[1] > 1 else y,
        )
        print(f"[train] train: {len(train_idx)}, val: {len(val_idx)}（数据按需从 mmap 读取）")

        train_ds = _RoiTensorDataset(X, y, train_idx)
        val_ds = _RoiTensorDataset(X, y, val_idx)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=False)

        y_train = y[train_idx]
        pos_counts = y_train.sum(axis=0)
        neg_counts = len(train_idx) - pos_counts
        pos_weight = torch.tensor(
            [neg_counts[i] / max(pos_counts[i], 1) for i in range(len(classes))],
            dtype=torch.float32,
        ).to(model.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            list(model.backbone.parameters()) + list(model.classifier.parameters()),
            lr=lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        mode = best_metric_mode(best_metric)
        best_value: float | None = None
        best_epoch: int | None = None
        best_val_loss: float | None = None
        # FeatureFusionModel 不是 nn.Module：state_dict() 是
        # {backbone_state, classifier_state} 两层嵌套 dict，必须整体拷贝——逐张量 .detach()
        # 会在嵌套 dict 上抛 AttributeError（既有 bug，验证指标链路时暴露）。
        best_state = copy.deepcopy(model.state_dict())
        # history 的验证指标键来自分类指标注册表（与正式评测同名同实现）；
        # val_acc 保留为历史别名（= val_exact_match，样本级全标签匹配率）。
        history = {
            "train_loss": [], "val_loss": [], "val_acc": [],
            **{key: [] for key in classification_training_keys()},
        }

        for epoch in range(1, epochs + 1):
            model.backbone.train()
            model.classifier.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(model.device), by.to(model.device)
                optimizer.zero_grad()
                feat = model.backbone(bx)
                loss = criterion(model.forward_from_features(feat), by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(bx)
            train_loss /= max(len(train_ds), 1)

            model.backbone.eval()
            model.classifier.eval()
            val_loss = 0.0
            batch_counts = []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(model.device), by.to(model.device)
                    feat = model.backbone(bx)
                    logits = model.forward_from_features(feat)
                    val_loss += criterion(logits, by).item() * len(bx)
                    # 判正走注册表的唯一阈值入口，训练期与评测期不会各写一份 0.5。
                    batch_counts.append(
                        confusion_counts(
                            decide(torch.sigmoid(logits).cpu().numpy()),
                            by.cpu().numpy(),
                        )
                    )
            val_loss /= max(len(val_ds), 1)
            val_metrics = metrics_from_counts(merge_counts(batch_counts), classes)
            # 键 → 取值位置由注册表给出（value_path），这里不再手抄对照表。
            epoch_values = {"val_loss": val_loss, **classification_training_values(val_metrics)}
            val_acc = epoch_values["val_exact_match"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)  # 旧别名，与 val_exact_match 同值
            for key in classification_training_keys():
                history[key].append(epoch_values[key])
            scheduler.step()

            if epoch % 5 == 0 or epoch == epochs:
                print(f"  epoch {epoch:3d}/{epochs}: train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                      f"val_f1={val_metrics['micro']['f1']:.4f}")
            if metric_improved(epoch_values[best_metric], best_value, mode):
                best_value = epoch_values[best_metric]
                best_epoch = epoch
                # best_val_loss 的语义是"被选中那个 epoch 的 val_loss"（默认按 val_loss 选点时即最小值）。
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

        info = {
            "name": best_metric,
            "mode": mode,
            "value": best_value,
            "epoch": best_epoch,
            "val_loss": best_val_loss,
        }
        print(f"[train] 最佳 {best_metric}={best_value} (epoch {best_epoch})")
        return history, best_state, info

    def predict(self, cfg: dict, ckpt: str, device) -> PredictionOutput:
        """加载 checkpoint，对 ROI 数据推理，返回含逐类 P/R/F1 的事实结果。"""

        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        classes = list(data_cfg["classes"])

        state, meta = load_checkpoint(
            ckpt,
            expected={"type": model_cfg["type"]},
            map_location="cpu",
            require_meta_schema=cfg.get("evaluation", {}).get("mode", "formal") == "formal",
        )
        actual_classes = meta.get("classes") or classes
        # 结构超参必须取自 checkpoint 绑定 meta（训练时写入的 model 段），否则 hidden_dim /
        # backbone 与权重不一致会在 load_state_dict 处 shape mismatch（既有 bug：此前只传
        # num_classes/backbone，hidden_dim 回落到默认值，导致非默认 hidden_dim 的权重无法评估）。
        train_model_cfg = dict(meta.get("model") or {})
        model = FeatureFusionModel(
            num_classes=len(actual_classes),
            backbone_name=meta.get("backbone", model_cfg.get("backbone", "resnet50")),
            freeze_backbone=bool(train_model_cfg.get("freeze_backbone", model_cfg.get("freeze_backbone", False))),
            hidden_dim=int(train_model_cfg.get("hidden_dim", model_cfg.get("hidden_dim", 256))),
            dropout=float(train_model_cfg.get("dropout", model_cfg.get("dropout", 0.3))),
        )
        model.load_state_dict(state)
        model.to_device(str(device))

        ds_base = self._dataset_dir(data_cfg)
        try:
            X, y, loaded_classes = load_dataset(actual_classes, ds_base)
        except FileNotFoundError:
            group_dir = Path(data_cfg["group_dir"])
            if not group_dir.is_absolute():
                group_dir = Path(__file__).resolve().parents[3] / group_dir
            X, y, loaded_classes, _ = build_roi_dataset(
                group_dir,
                actual_classes,
                roi_size=int(meta.get("input_size", model_cfg.get("roi_size", 224))),
                neg_ratio=float(data_cfg.get("neg_ratio", 1.0)),
            )
        actual_classes = loaded_classes

        native = self._evaluate(model, X, y, actual_classes, device)
        return PredictionOutput(
            model_type=model_cfg["type"],
            model_id=f"feature_fusion-{model_cfg.get('backbone', 'resnet50')}",
            pipeline=self.pipeline_name,
            checkpoint=str(ckpt),
            dataset=data_cfg.get("name", "-".join(actual_classes)),
            predictions={},
            labels={i: name for i, name in enumerate(actual_classes)},
            feature_schema={"modality": "roi_image", "roi_size": model.input_size},
            inference_semantics={"mode": "single_roi", "stateless": True},
            num_params=sum(p.numel() for p in model.backbone.parameters())
            + sum(p.numel() for p in model.classifier.parameters()),
            native_metrics=native,
            metadata={"dataset_dir": str(ds_base)},
            errors=[],
        )

    def _evaluate(self, model, X, y, classes, device, batch_size: int = 32) -> dict:
        """多标签 P/R/F1 与 exact-match 计算：复用 ``core/metrics.py`` 注册表的唯一实现。

        返回普通 dict（``per_class`` / ``micro`` / ``exact_match`` / ``labels``）供 benchmark
        evaluator 翻译成三态指标；训练期 validation 走的是同一套计数与换算函数。
        """

        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        loader = DataLoader(
            _RoiTensorDataset(X, y, np.arange(len(y))),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )

        model.backbone.eval()
        model.classifier.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for bx, by in loader:
                bx = bx.to(model.device)
                feat = model.backbone(bx)
                logits = model.forward_from_features(feat)
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(by.numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        binary_preds = decide(all_preds)

        return multilabel_metrics(binary_preds, all_labels, classes)
