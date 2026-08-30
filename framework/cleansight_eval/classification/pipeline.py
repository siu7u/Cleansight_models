"""ROI 图像分类流水线（roi_classification）。

特征融合模型的标准 Pipeline 实现：训练消费 YOLO 分组数据集的 GT 框裁剪 ROI 数据，
训练 CNN backbone + MLP 多标签头；预测对 ROI 图像块逐类输出 sigmoid 概率。
checkpoint 复用 ``core.checkpoint.save_checkpoint`` 的绑定 meta 契约。

torch/torchvision 为重依赖，全部在方法内部 import（与 detection/yolo.py 惯例一致），
使仅做配置/边界校验的场景无需安装它们。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.checkpoint import load_checkpoint, save_checkpoint
from ..core.environment import now_stamp, set_seed
from ..core.execution import PredictionOutput
from ..core.pipeline import Pipeline
from ..core.run import RunContext
from .data import build_roi_dataset, load_dataset, save_dataset
from .model import BACKBONE_CONFIGS, FeatureFusionModel


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

    def _dataset_dir(self, data_cfg: dict) -> Path:
        from ..core.run import RunContext  # noqa: F401  (保持导入一致性)

        base = Path(data_cfg.get("dataset_dir") or "runs/feature_fusion/datasets")
        if not base.is_absolute():
            base = Path(__file__).resolve().parents[3] / base
        return base

    def train(self, cfg: dict, runs_dir: str, seed: int, device, run_id: str | None = None) -> str:
        set_seed(seed)
        model_cfg = cfg["model"]
        data_cfg = cfg["data"]
        train_cfg = cfg.get("train", {})
        classes = list(data_cfg["classes"])
        group_dir = Path(data_cfg["group_dir"])
        if not group_dir.is_absolute():
            group_dir = Path(__file__).resolve().parents[3] / group_dir

        run = RunContext(runs_dir, label="classification", run_id=run_id)
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

            history, best_state, best_val = self._fit(
                model, X, y, classes,
                epochs=int(train_cfg.get("epochs", 50)),
                batch_size=int(train_cfg.get("batch_size", 32)),
                lr=float(train_cfg.get("lr", 1e-3)),
                val_split=float(data_cfg.get("val_split", 0.2)),
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
                "best_val_loss": best_val,
            }
            save_checkpoint(ckpt, model.state_dict(), meta)

            history_path = run.dir / "history.json"
            history_path.write_text(json.dumps(history, indent=2, default=str),
                                    encoding="utf-8")
            run.write_status("succeeded", best_checkpoint=str(ckpt),
                             best_val_loss=best_val, history=str(history_path))
            print(f"[train] run_dir={run.dir}")
            print(f"[train] checkpoint={ckpt}")
            return str(ckpt)
        except Exception as exc:
            run.write_exception_status(exc)
            raise

    def _fit(self, model, X, y, classes, *, epochs, batch_size, lr, val_split):
        """训练循环，返回 (history, best_state_dict, best_val_loss)。"""

        import numpy as np
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42,
            stratify=y.any(axis=1) if y.shape[1] > 1 else y,
        )
        print(f"[train] train: {len(X_train)}, val: {len(X_val)}")

        def _to_rgb(tensor):
            return tensor[:, [2, 1, 0], :, :]  # BGR -> RGB

        train_ds = TensorDataset(
            _to_rgb(torch.from_numpy(X_train).float().permute(0, 3, 1, 2)),
            torch.from_numpy(y_train).float(),
        )
        val_ds = TensorDataset(
            _to_rgb(torch.from_numpy(X_val).float().permute(0, 3, 1, 2)),
            torch.from_numpy(y_val).float(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=False)

        pos_counts = y_train.sum(axis=0)
        neg_counts = len(y_train) - pos_counts
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

        best_val_loss = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

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
            correct = 0
            total = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(model.device), by.to(model.device)
                    feat = model.backbone(bx)
                    logits = model.forward_from_features(feat)
                    val_loss += criterion(logits, by).item() * len(bx)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct += (preds == by).all(dim=1).sum().item()
                    total += len(bx)
            val_loss /= max(len(val_ds), 1)
            val_acc = correct / max(total, 1)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            scheduler.step()

            if epoch % 5 == 0 or epoch == epochs:
                print(f"  epoch {epoch:3d}/{epochs}: train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[train] 最佳 val_loss={best_val_loss:.4f}")
        return history, best_state, best_val_loss

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
        model = FeatureFusionModel(
            num_classes=len(actual_classes),
            backbone_name=meta.get("backbone", model_cfg.get("backbone", "resnet50")),
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
        """多标签 P/R/F1 计算，返回普通 dict 供 benchmark evaluator 翻译。"""

        import numpy as np
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        ds = TensorDataset(
            torch.from_numpy(X).float().permute(0, 3, 1, 2)[:, [2, 1, 0], :, :],
            torch.from_numpy(y).float(),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

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
        binary_preds = (all_preds > 0.5).astype(np.float32)

        per_class = {}
        for i, cls_name in enumerate(classes):
            tp = ((binary_preds[:, i] == 1) & (all_labels[:, i] == 1)).sum()
            fp = ((binary_preds[:, i] == 1) & (all_labels[:, i] == 0)).sum()
            fn = ((binary_preds[:, i] == 0) & (all_labels[:, i] == 1)).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            per_class[cls_name] = {
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "support": int(all_labels[:, i].sum()),
            }

        tp_all = ((binary_preds == 1) & (all_labels == 1)).sum()
        fp_all = ((binary_preds == 1) & (all_labels == 0)).sum()
        fn_all = ((binary_preds == 0) & (all_labels == 1)).sum()
        micro_p = tp_all / max(tp_all + fp_all, 1)
        micro_r = tp_all / max(tp_all + fn_all, 1)
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)
        acc = float(((binary_preds == all_labels).all(axis=1)).mean())

        return {
            "per_class": per_class,
            "micro": {
                "precision": round(float(micro_p), 4),
                "recall": round(float(micro_r), 4),
                "f1": round(float(micro_f1), 4),
            },
            "exact_match": round(acc, 4),
            "labels": {i: name for i, name in enumerate(classes)},
        }
