#!/usr/bin/env python3
"""在 train split 的全部帧 embedding 上拟合 PCA-80（S2/S3 管线第 3 步）。

见 docs/FEATURE_SCHEME_S2_S3_SPEC.md §3.3。读
``<embedding_dir>/train/<视频>.mp4.npy``（[T, 512]），零均值 PCA（numpy SVD），
写 `<embedding_dir>/pca80.npz`（mean/components/meta）。只在 train 上拟合——val
推理仅使用该固定变换，保证因果与可复现。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROJ_DIM = 80


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding-dir", required=True)
    ap.add_argument("--dim", type=int, default=PROJ_DIM)
    args = ap.parse_args()

    emb_dir = Path(args.embedding_dir)
    npys = sorted((emb_dir / "train").glob("*.npy"))
    if not npys:
        raise SystemExit(f"train split 没有 embedding: {emb_dir / 'train'}")
    mats = [np.load(p) for p in npys]
    x = np.concatenate(mats, axis=0).astype(np.float64)
    mean = x.mean(axis=0)
    xc = x - mean
    # SVD 即主成分：components 行 = 主成分方向
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    components = vt[: args.dim]
    evr = float((vt[: args.dim] ** 2).sum() / (vt ** 2).sum())
    out = emb_dir / "pca80.npz" if args.dim == PROJ_DIM else emb_dir / f"pca{args.dim}.npz"
    np.savez(
        out,
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
    )
    meta = {
        "dim": args.dim,
        "explained_variance_ratio": evr,
        "fit_frames": int(x.shape[0]),
        "fit_videos": len(npys),
        "fit_split": "train",
        "fitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"pca -> {out}  evr={evr:.4f}  frames={x.shape[0]} videos={len(npys)}")


if __name__ == "__main__":
    main()
