"""时序模型注册表（两条时序流水线共用）。

模型退化为可替换的纯 ``nn.Module`` 组件：只提供网络结构与「是否因果」标志。监督口径与
推理方式由流水线拥有，不写在模型里。新增架构（Transformer / causal-TCN…）在此登记一行
即可被两条时序流水线复用——``causal=True`` 才允许进滑窗流水线。

配置以 ``model.type`` 引用（如 ``type: gru``），规模等超参走同一 ``model`` 段。
"""

from __future__ import annotations

import torch.nn as nn

from .clean_offline import CleanASFormer, CleanBiGRU, CleanMSTCNBiLSTM
from .gru import GRUClassifier
from .mstcn import MSTCN
from .mstcn2 import MSTCN2
from .transformer import TransformerClassifier


def _build_gru(cfg: dict) -> nn.Module:
    return GRUClassifier(
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
        hidden=cfg.get("hidden", 128),
        num_layers=cfg.get("num_layers", 3),
    )


def _build_mstcn(cfg: dict) -> nn.Module:
    return MSTCN(
        in_dim=cfg["input_dim"],
        classes=cfg["num_classes"],
        hidden=cfg.get("hidden", 32),
    )


def _build_mstcn2(cfg: dict) -> nn.Module:
    return MSTCN2(
        in_dim=cfg["input_dim"],
        classes=cfg["num_classes"],
        hidden=cfg.get("hidden", 64),
        num_stages=cfg.get("num_stages", 4),
        num_layers=cfg.get("num_layers", 10),
        dropout=cfg.get("dropout", 0.3),
        tmse_weight=cfg.get("tmse_weight", 0.15),
        tmse_clip=cfg.get("tmse_clip", 4.0),
    )


def _build_transformer(cfg: dict) -> nn.Module:
    return TransformerClassifier(
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
        d_model=cfg.get("d_model", 64),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
        dim_feedforward=cfg.get("dim_feedforward", 128),
        dropout=cfg.get("dropout", 0.1),
        max_len=cfg.get("max_len", 2048),
    )


def _build_clean_asformer(cfg: dict) -> nn.Module:
    return CleanASFormer(
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
        hidden=cfg.get("hidden", 64),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 4),
        dropout=cfg.get("dropout", 0.15),
    )


def _build_clean_bigru(cfg: dict) -> nn.Module:
    return CleanBiGRU(
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
        hidden=cfg.get("hidden", 64),
        num_layers=cfg.get("num_layers", 3),
        dropout=cfg.get("dropout", 0.15),
    )


def _build_clean_mstcn_bilstm(cfg: dict) -> nn.Module:
    return CleanMSTCNBiLSTM(
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
        hidden=cfg.get("hidden", 64),
        lstm_layers=cfg.get("lstm_layers", 2),
        tcn_layers=cfg.get("tcn_layers", 6),
        refine_stages=cfg.get("refine_stages", 2),
        dropout=cfg.get("dropout", 0.15),
    )


# type -> {build: cfg->nn.Module, causal: bool}
_MODELS = {
    "clean_asformer": {"build": _build_clean_asformer, "causal": False},
    "clean_bigru": {"build": _build_clean_bigru, "causal": False},
    "clean_mstcn_bilstm": {"build": _build_clean_mstcn_bilstm, "causal": False},
    "gru": {"build": _build_gru, "causal": True},
    "mstcn": {"build": _build_mstcn, "causal": False},
    "mstcn2": {"build": _build_mstcn2, "causal": False},
    "transformer": {"build": _build_transformer, "causal": False},
}


def build_model(model_cfg: dict) -> nn.Module:
    """按 ``model_cfg["type"]`` 构造网络。"""
    t = model_cfg.get("type")
    if t not in _MODELS:
        raise KeyError(f"未注册的时序模型: {t!r}；已注册: {sorted(_MODELS)}")
    return _MODELS[t]["build"](model_cfg)


def is_causal(model_type: str) -> bool:
    """该模型是否因果（可用于滑窗流式推理）。"""
    if model_type not in _MODELS:
        raise KeyError(f"未注册的时序模型: {model_type!r}；已注册: {sorted(_MODELS)}")
    return _MODELS[model_type]["causal"]
