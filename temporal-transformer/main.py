import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

from dataloader import build_dataset
from model import *
from util import (
    load_mappings, 
    load_features, 
    load_truths,
    compute_class_weights,
    get_current_timestamp,
    causal_decision,
    f_score,
    edit_score,
    plot_temporal_results
)
from lab import load_data_json


MAPPING_PATH = "data/Endo_Project/mapping.txt"
FEATURES_DIR = "data/Endo_Project/features"
TRUTHS_DIR = "data/Endo_Project/groundTruth"

video_names = [
    "export1", "export2", "export3", "export4",
    "export5", "export6", "export7-480p", "export8",
    "export9", "export10", "export11", "export12", 
    "export13", "export14", "export15-480P", "export16-480P", 
    "export17", "export18", "export19", "export20",
]

TRAIN_IDX = list(range(0, 16))
TEST_IDX = list(range(16, 20)) # [17]

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(
    model, 
    dataloader, 
    optimizer, 
    criterion, 
    epochs, 
    save_prefix,
    verbose=False,
    auto_save=False,
):
    model.train()

    for epoch in tqdm(range(1, epochs + 1)):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)                 # (B, w, C)
            last_logits = logits[:, -1, :]    # (B, C)
    
            loss = criterion(last_logits, y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:3d}  Loss {loss.item():.4f}")

        if auto_save and epoch % 5 == 0:
            import os
            os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
            torch.save(model.state_dict(), f"{save_prefix}-{epoch}-{get_current_timestamp()}.pt")

def eval(
    model,
    dataloader,
    args,
    model_path,
    visualize=False,
):
    model.eval()

    video_preds = []
    video_gts = []

    for ds in tqdm(dataloader.dataset.datasets):
        T = ds.x.shape[0]
        window = ds.w
        preds = np.zeros(T, dtype=np.int64)

        # 补全前 window-1 帧
        idle_id = 0
        preds[:window-1] = idle_id

        # ===== 决策状态 =====
        pending = None     # 待确认状态
        stable = idle_id   # 已确认状态（连续帧）
        count = 0
        
        with torch.no_grad():
            for i in range(len(ds)):
                x, _ = ds[i]
                x = x.unsqueeze(0).to(device)

                logits = model(x)            # (1, w, C)
                last = logits[0, -1]

                pending, stable, count = causal_decision(last, pending, stable, count)
                preds[i + window - 1] = stable

        video_preds.append(preds)
        video_gts.append(ds.y.numpy())

    # ===== 合并所有视频 =====
    all_preds = np.concatenate(video_preds)
    all_gts   = np.concatenate(video_gts)

    mappings = load_mappings(MAPPING_PATH)
    idx_to_action = {v : k for k, v in mappings.items()}
    pred_labels = [idx_to_action[p] for p in all_preds]
    gt_labels   = [idx_to_action[g] for g in all_gts]

    # Accuracy
    correct = sum(p == g for p, g in zip(all_preds, all_gts))
    acc = 100.0 * correct / len(all_preds)

    # Edit score
    edit = edit_score(pred_labels, gt_labels)

    # F1 scores
    overlap = [0.1, 0.25, 0.5]
    tp = np.zeros(3)
    fp = np.zeros(3)
    fn = np.zeros(3)

    for i, o in enumerate(overlap):
        tp1, fp1, fn1 = f_score(pred_labels, gt_labels, o)
        tp[i] += tp1
        fp[i] += fp1
        fn[i] += fn1

    f1_scores = {}
    for i, o in enumerate(overlap):
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall    = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        f1_scores[o] = f1 * 100

    result = {
        "timestamp": get_current_timestamp(),
        "model": args.model,
        "path": model_path,
        "num_params": sum(p.numel() for p in model.parameters()),
        "acc": round(acc, 2),
        "edit": round(edit, 2),
        "f1": {k: round(v, 2) for k, v in f1_scores.items()}
    }

    import json
    print(json.dumps(result, indent=4, ensure_ascii=False))

    if visualize and args.output_dir:
        import os
        os.makedirs(f"{args.output_dir}", exist_ok=True)
        plot_temporal_results(
            pred_labels, gt_labels,
            metadata=result,
            output_path=f"{args.output_dir}/{args.model}-{get_current_timestamp()}.png",
        )

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Causal Action Recognition")

    # === mode ===
    parser.add_argument("--mode", type=str, default="full", help="模式选择",
                        choices=["full", "train", "eval"])
    # === model ===
    parser.add_argument("--model", type=str, default="gru", help="模型类型",
                        choices=["gru", "tcn", "transformer"])
    parser.add_argument("--input_dim", type=int, default=20, help="特征维度")
    parser.add_argument("--num_classes", type=int, default=3, help="类别数")
    parser.add_argument("--resume", type=str, default=None, help="从指定检查点恢复训练")

    # === train ===
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--window", type=int, default=64, help="历史帧数")
    parser.add_argument("--verbose", action="store_true", help="是否打印训练损失")

    # === persist ===
    parser.add_argument("--auto_save", action="store_true", help="是否保存中间结果")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="检查点保存目录")
    parser.add_argument("--export_dir", type=str, default="weights", help="模型权重保存目录")

    # === eval ===
    parser.add_argument("--visualize", action="store_true", help="是否生成可视化结果")
    parser.add_argument("--output_dir", type=str, default="results", help="图标结果目录")

    args = parser.parse_args()

    # ===== Model =====
    if args.model == "gru":
        model = GRUClassifier(args.input_dim, args.num_classes)
    elif args.model == "tcn":
        model = TCNClassifier(args.input_dim, args.num_classes)
    elif args.model == "transformer":
        model = TransformerClassifier(args.input_dim, args.num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model_path = None
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        model_path = args.resume
    model = model.to(device)

    # ===== Dataset =====
    mappings = load_mappings(MAPPING_PATH)

    features, truths = [], []
    # 暂时复用 MS-TCN2 训练数据
    for name in video_names:
        features.append(load_features(FEATURES_DIR, name))
        truths.append(load_truths(TRUTHS_DIR, name, mappings=mappings))

    # # Label Studio 数据加载示例
    # for id in [50, 51, 52, 53, 54, 55, 56, 58, 59]:
    #     print(id)
    #     feats, trs = load_data_json("path to LS json", id)
    #     features.append(feats)
    #     truths.append([mappings[tr] for tr in trs])

    # 暂时用下标指定数据集，之后可以调整
    train_dataset = build_dataset(features, truths, idx=TRAIN_IDX, window=args.window)
    test_dataset = build_dataset(features, truths, idx=TEST_IDX, window=args.window)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # ===== Training =====
    if args.mode in ["full", "train"]:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        # criterion = nn.CrossEntropyLoss()
        train_weights = compute_class_weights(train_loader)

        criterion = nn.CrossEntropyLoss(weight=torch.tensor(
            [train_weights[i] for i in sorted(train_weights.keys())],
            dtype=torch.float32
        ).to(device))

        train(model, train_loader, optimizer, criterion, args.epochs, 
              verbose=args.verbose, auto_save=args.auto_save, 
              save_prefix=f"{args.save_dir}/{args.model}-w{args.window}")

        if args.export_dir:
            import os
            os.makedirs(args.export_dir, exist_ok=True)
            model_path = f"{args.export_dir}/{args.model}-final-{get_current_timestamp()}.pt"
            torch.save(model.state_dict(), model_path)

    # ===== Evaluation =====
    if args.mode in ["full", "eval"]:
        eval(model, test_loader, visualize=args.visualize, args=args, model_path=model_path)
