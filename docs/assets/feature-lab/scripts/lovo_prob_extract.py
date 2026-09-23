# -*- coding: utf-8 -*-
"""LOVO nodep 各折 held-out 视频的逐帧 softmax 概率提取（WSL，存 npz 供画图）。

复用框架的 load_checkpoint/build_model/load_split/SlidingWindowDataset，
推理语义与训练期 _evaluate_sliding_window 一致（逐窗末帧输出，不做 causal_decision 平滑，
保留原始概率）。输出：tmp/report_charts/lovo_probs/foldXX.npz（probs[T,6] / gt[T] / names[6]）。
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/mnt/e/曦源/Cleansight_models')
sys.path.insert(0, str(ROOT))

from framework.cleansight_eval.core.config import load_config
from framework.cleansight_eval.core.checkpoint import load_checkpoint
from framework.cleansight_eval.temporal.models import build_model
from framework.cleansight_eval.temporal.data import load_split
from framework.cleansight_eval.temporal.sliding_window_pipeline import SlidingWindowDataset

OUT = ROOT / 'tmp/report_charts/lovo_probs'
OUT.mkdir(parents=True, exist_ok=True)

CFG_PATH = ROOT / 'tmp/gru-nodep-wsl.yaml'
RUNS = ROOT / 'runs/lovo'
FOLDS = ROOT / 'tmp/lovo'

device = torch.device('cpu')

# progress.txt: fold done <video> <run_dir>
folds = {}
for line in (RUNS / 'progress.txt').read_text(encoding='utf-8').splitlines():
    parts = line.split()
    if len(parts) >= 4 and parts[1] == 'done':
        folds[parts[0]] = (parts[2], Path(parts[3].replace('/mnt/e', '/mnt/e')))

for fold in sorted(folds):
    video, run_dir = folds[fold]
    if (OUT / f'{fold}.npz').exists():
        print(f'{fold}: already done, skip')
        continue
    ckpt = run_dir / 'checkpoints/best.pt'
    if not ckpt.exists():
        print(f'{fold}: MISSING ckpt {ckpt}')
        continue
    cfg = load_config(str(CFG_PATH))
    cfg['data']['root'] = str(FOLDS / fold)   # fold 目录：val/ 即 held-out 视频
    cfg['data']['split_eval'] = 'val'

    model_cfg = cfg['model']
    expected = {'type': model_cfg['type'], 'input_dim': model_cfg['input_dim'],
                'num_classes': model_cfg['num_classes']}
    state_dict, meta = load_checkpoint(str(ckpt), expected=expected, map_location=device,
                                       require_meta_schema=False)
    model = build_model(meta['model']).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    window = meta.get('window') or cfg['train'].get('window', 16)

    features, truths, id2name = load_split(cfg['data'], 'val', window=window,
                                           feature_schema=cfg.get('feature_schema'))
    names = [id2name[i] for i in range(len(id2name))]
    for vi in range(len(features)):
        ds = SlidingWindowDataset(features[vi], truths[vi], window)
        T = ds.x.shape[0]
        probs = np.zeros((T, len(names)), dtype=np.float32)
        with torch.no_grad():
            for i in range(len(ds)):
                x = ds[i][0].unsqueeze(0).to(device)      # [1, window, F]
                logits = model(x)[0, -1]                   # 末帧 logits
                probs[i + window - 1] = torch.softmax(logits, dim=-1).cpu().numpy()
        # 冷启动前 window-1 帧概率置为 idle 先验（与 predict() 冷启动语义一致）
        if len(names) > 0 and 'idle' in names:
            probs[:window - 1, names.index('idle')] = 1.0
        np.savez_compressed(OUT / f'{fold}.npz', probs=probs, gt=ds.y.numpy(),
                            names=np.array(names), video=video, window=window,
                            run_dir=str(run_dir))
        print(f'{fold}: video={video[:20]}... T={T} classes={len(names)} saved')
print('ALL DONE')
