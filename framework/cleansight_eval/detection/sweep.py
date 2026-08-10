"""检测域优化实验编排（sweep）。

提供多预设 / grid 搜索的 YOLO 训练实验编排：复用 ``YoloAdapter.train/val`` 与
``core.checkpoint.write_meta``，不直接 import ultralytics。每个实验产出整体 + 逐类
P/R/mAP50 指标并写入 ``runs/optimize_reports/`` 下的 JSON 与 Markdown 报告。

CLI 入口见 ``cli/sweep.py``（``python -m framework.cleansight_eval.cli.sweep``）。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .yolo import get_adapter

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_BASE = REPO_ROOT / "datasets" / "cleansight-yolo"
RUNS_BASE = REPO_ROOT / "runs" / "cleansight-yolo"
REPORTS_DIR = REPO_ROOT / "runs" / "optimize_reports"

VALID_GROUPS = ("group1_large", "group2_small")

# ── SMOKE 探针默认值 ─────────────────────────────────────────
# 探针模式只用于预设方向对比：epochs 截断、patience 收紧、fraction 子采样，
# 结果一律标记 smoke，禁止当作正式指标引用。
SMOKE_EPOCHS = 15      # 探针模式最大 epoch 数
SMOKE_PATIENCE = 5     # 探针模式早停耐心
SMOKE_FRACTION = 0.2   # 探针模式每 epoch 数据采样比例

# ── 预定义预设 ──────────────────────────────────────────────
PRESETS: Dict[str, dict] = {
    # === group1_large 预设 ===
    "large_baseline": {
        "description": "large 组基线: yolo11n, imgsz 640, 默认增强",
        "model": "yolo11n.pt",
        "imgsz": 640,
        "epochs": 150,
        "batch": 16,
        "patience": 30,
        "augment": "default",
        "label_smoothing": 0.0,
        "cos_lr": False,
        "close_mosaic": 10,
    },
    "large_s": {
        "description": "large 组: yolo11s, imgsz 640",
        "model": "yolo11s.pt",
        "imgsz": 640,
        "epochs": 150,
        "batch": 16,
        "patience": 30,
        "augment": "default",
    },
    "large_m": {
        "description": "large 组: yolo11m, imgsz 640",
        "model": "yolo11m.pt",
        "imgsz": 640,
        "epochs": 150,
        "batch": 12,
        "patience": 30,
        "augment": "default",
    },
    "large_s_960": {
        "description": "large 组: yolo11s, imgsz 960, 增强",
        "model": "yolo11s.pt",
        "imgsz": 960,
        "epochs": 150,
        "batch": 12,
        "patience": 30,
        "augment": "strong",
    },
    "large_s_1280": {
        "description": "large 组: yolo11s, imgsz 1280",
        "model": "yolo11s.pt",
        "imgsz": 1280,
        "epochs": 150,
        "batch": 8,
        "patience": 30,
        "augment": "default",
    },
    "large_m_960": {
        "description": "large 组: yolo11m, imgsz 960, 强增强+cos",
        "model": "yolo11m.pt",
        "imgsz": 960,
        "epochs": 200,
        "batch": 8,
        "patience": 40,
        "augment": "strong",
        "cos_lr": True,
        "label_smoothing": 0.1,
    },
    "large_s_freeze": {
        "description": "large 组: yolo11s, freeze backbone 10 层",
        "model": "yolo11s.pt",
        "imgsz": 640,
        "epochs": 150,
        "batch": 16,
        "patience": 30,
        "augment": "default",
        "freeze": 10,
    },
    # === group2_small 预设 ===
    "small_baseline": {
        "description": "small 组基线: yolo11n, imgsz 640",
        "model": "yolo11n.pt",
        "imgsz": 640,
        "epochs": 150,
        "batch": 16,
        "patience": 30,
        "augment": "default",
    },
    "small_s_960": {
        "description": "small 组: yolo11s, imgsz 960",
        "model": "yolo11s.pt",
        "imgsz": 960,
        "epochs": 150,
        "batch": 12,
        "patience": 30,
        "augment": "strong",
    },
    "small_n_1280_p2": {
        "description": "small 组: yolo11n, imgsz 1280, P2 head (小目标)",
        "model": "yolo11n.pt",
        "imgsz": 1280,
        "epochs": 200,
        "batch": 8,
        "patience": 40,
        "augment": "strong",
        "cos_lr": True,
    },
    "small_s_1280_p2": {
        "description": "small 组: yolo11s, imgsz 1280, P2 head + class weights",
        "model": "yolo11s.pt",
        "imgsz": 1280,
        "epochs": 200,
        "batch": 8,
        "patience": 40,
        "augment": "strong",
        "cos_lr": True,
    },
    "small_m_1280": {
        "description": "small 组: yolo11m, imgsz 1280, 强增强+cos+ls",
        "model": "yolo11m.pt",
        "imgsz": 1280,
        "epochs": 200,
        "batch": 6,
        "patience": 50,
        "augment": "strong",
        "cos_lr": True,
        "label_smoothing": 0.1,
    },
    "small_s_copy_paste": {
        "description": "small 组: yolo11s, copy_paste 增强(稀有类)",
        "model": "yolo11s.pt",
        "imgsz": 960,
        "epochs": 200,
        "batch": 12,
        "patience": 40,
        "augment": "copy_paste",
        "cos_lr": True,
    },
}

# ── Grid 维度定义 ──────────────────────────────────────────
GRID_DIMS = {
    "models": {
        "n": {"model": "yolo11n.pt", "batch_base": 16},
        "s": {"model": "yolo11s.pt", "batch_base": 16},
        "m": {"model": "yolo11m.pt", "batch_base": 12},
    },
    "resolutions": {
        "640": 640,
        "960": 960,
        "1280": 1280,
    },
    "augments": {
        "default": "default",
        "strong": "strong",
    },
}


def get_augment_params(name: str, imgsz: int) -> dict:
    """返回 ultralytics 增强超参 dict（default / strong / copy_paste）。"""

    base = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
    }
    if name == "default":
        return base
    if name == "strong":
        return {
            **base,
            "hsv_h": 0.02,
            "hsv_s": 0.8,
            "hsv_v": 0.5,
            "scale": 0.7,
            "translate": 0.2,
            "shear": 2.0,
            "mosaic": 1.0,
            "mixup": 0.15,
            "copy_paste": 0.0,
        }
    if name == "copy_paste":
        return {
            **base,
            "hsv_h": 0.02,
            "hsv_s": 0.8,
            "hsv_v": 0.5,
            "scale": 0.7,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.3,
        }
    return base


def adjust_batch_for_imgsz(base_batch: int, base_imgsz: int, target_imgsz: int) -> int:
    """按分辨率等比缩放 batch size（保持显存大致恒定）。"""

    ratio = (base_imgsz / target_imgsz) ** 2
    return max(2, int(base_batch * ratio))


def build_experiment_name(preset_name: str, group: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"opt-{group}-{preset_name}-{ts}"


def run_experiment(
    group: str,
    preset_name: str,
    cfg: dict,
    dry_run: bool = False,
    smoke: bool = False,
    device: Optional[str] = None,
) -> dict:
    """执行单个实验: 训练 + val 评测，返回指标 dict（复用 YoloAdapter）。

    smoke=True 为快速探针模式：epochs 截断到 SMOKE_EPOCHS、patience 收紧到
    SMOKE_PATIENCE，并按 SMOKE_FRACTION 子采样数据；结果标记 smoke，只用于
    预设方向对比，不代表正式指标。device 显式传入时优先于 cfg 里的 device 键。
    """

    group_dir = DATASET_BASE / group
    data_yaml = group_dir / "data.yaml"
    if not data_yaml.is_file():
        return {"error": f"数据集缺失: {data_yaml}"}

    name = build_experiment_name(preset_name, group)
    project = str(RUNS_BASE.resolve())
    cfg = dict(cfg)
    if smoke:
        cfg["epochs"] = min(cfg.get("epochs", 150), SMOKE_EPOCHS)
        cfg["patience"] = SMOKE_PATIENCE
    model_file = cfg.get("model", "yolo11n.pt")
    # run_experiment 稍后会 chdir 到分组目录，裸权重名在那之后失效（ultralytics
    # 找不到会转去 GitHub 下载）；仓库根目录存在同名权重时提前解析为绝对路径。
    if not Path(model_file).is_absolute() and Path(model_file).name == model_file:
        local_weight = REPO_ROOT / model_file
        if local_weight.is_file():
            model_file = str(local_weight)
    imgsz = cfg.get("imgsz", 640)
    epochs = cfg.get("epochs", 150)
    patience = cfg.get("patience", 30)
    augment_name = cfg.get("augment", "default")
    augment = get_augment_params(augment_name, imgsz)
    cos_lr = cfg.get("cos_lr", False)
    label_smoothing = cfg.get("label_smoothing", 0.0)
    close_mosaic = cfg.get("close_mosaic", 10)
    freeze = cfg.get("freeze", None)
    batch = cfg.get("batch")
    if batch is None:
        batch = adjust_batch_for_imgsz(16, 640, imgsz)
    # 显式 device 参数（CLI --device）优先，其次 cfg（预设/测试可注入），最后 auto。
    device = device or cfg.get("device", "auto")

    smoke_tag = " [SMOKE 探针]" if smoke else ""
    print(f"\n{'='*60}")
    print(f"实验: {name}{smoke_tag}")
    print(f"  group={group}  preset={preset_name}  model={model_file}")
    print(f"  imgsz={imgsz}  epochs={epochs}  batch={batch}  patience={patience}"
          + (f"  fraction={SMOKE_FRACTION}" if smoke else ""))
    print(f"  augment={augment_name}  cos_lr={cos_lr}  label_smoothing={label_smoothing}")
    print(f"  freeze={freeze}  device={device}")
    print(f"{'='*60}")

    if dry_run:
        return {"name": name, "dry_run": True, "cfg": cfg, "smoke": smoke}

    if device == "auto":
        from ..core.environment import pick_device

        device = pick_device()

    # data.yaml 的 path 相对 cwd 解析，切到分组目录执行。
    os.chdir(group_dir)

    result = {
        "name": name,
        "group": group,
        "preset": preset_name,
        "cfg": cfg,
        "timestamp": datetime.now().isoformat(),
        "smoke": smoke,
    }

    adapter = get_adapter("yolo")
    train_kwargs = {
        "epochs": epochs,
        "batch": batch,
        "patience": patience,
        "cos_lr": cos_lr,
        "label_smoothing": label_smoothing,
        "close_mosaic": close_mosaic,
        **augment,
    }
    if smoke:
        train_kwargs["fraction"] = SMOKE_FRACTION
    if freeze is not None:
        train_kwargs["freeze"] = freeze

    try:
        best_pt, num_params, names, nc = adapter.train(
            weights=model_file,
            data_yaml=str(data_yaml),
            train_cfg=train_kwargs,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
        )
        result["best_pt"] = str(best_pt)
        result["num_params"] = num_params
        print(f"  训练完成: {best_pt}")
    except Exception as e:
        result["error"] = f"训练失败: {e}"
        print(f"  [ERROR] {e}")
        return result

    try:
        val = adapter.val(
            weights=str(best_pt),
            data_yaml=str(data_yaml),
            split="val",
            imgsz=imgsz,
            device=device,
            conf=0.001,
            iou=0.7,
            max_det=300,
            agnostic_nms=False,
        )
        result["val"] = {
            "map50": round(float(val["map50"]), 4),
            "map50_95": round(float(val["map50_95"]), 4),
            "precision": round(float(val["precision"]), 4),
            "recall": round(float(val["recall"]), 4),
            "per_class": val.get("per_class", {}),
        }
        print(f"  val: mAP50={result['val']['map50']:.4f}  "
              f"mAP50-95={result['val']['map50_95']:.4f}  "
              f"P={result['val']['precision']:.4f}  R={result['val']['recall']:.4f}")
    except Exception as e:
        result["val_error"] = str(e)
        print(f"  [VAL ERROR] {e}")

    return result


def run_grid(
    group: str,
    dims: List[str],
    base_cfg: Optional[dict] = None,
    dry_run: bool = False,
    smoke: bool = False,
    device: Optional[str] = None,
) -> List[dict]:
    """在指定维度上做 grid search（models / resolutions / augments）。

    smoke=True 时每个组合都按快速探针模式执行（见 ``run_experiment``）。
    """

    from itertools import product

    base = base_cfg or {
        "epochs": 150,
        "patience": 30,
        "augment": "default",
        "cos_lr": False,
        "label_smoothing": 0.0,
        "close_mosaic": 10,
    }

    selected_dims = {}
    for dim in dims:
        if dim == "models":
            selected_dims["models"] = list(GRID_DIMS["models"].items())
        elif dim == "resolutions":
            selected_dims["resolutions"] = list(GRID_DIMS["resolutions"].items())
        elif dim == "augments":
            selected_dims["augments"] = list(GRID_DIMS["augments"].items())

    if not selected_dims:
        print("[grid] 无有效维度，跳过")
        return []

    keys = list(selected_dims.keys())
    values = list(selected_dims.values())
    combinations = list(product(*values))

    print(f"\n[grid] 共 {len(combinations)} 个组合:")
    results = []
    for combo in combinations:
        cfg = dict(base)
        name_parts = []
        for k, (val_key, val_data) in zip(keys, combo):
            if k == "models":
                cfg.update(val_data)
                name_parts.append(f"m{val_key}")
            elif k == "resolutions":
                cfg["imgsz"] = val_data
                name_parts.append(f"r{val_key}")
            elif k == "augments":
                cfg["augment"] = val_data
                name_parts.append(f"a{val_key}")
        preset_name = "-".join(name_parts)
        print(f"  {preset_name}: model={cfg.get('model','?')} imgsz={cfg.get('imgsz','?')} "
              f"augment={cfg.get('augment','?')}")
        r = run_experiment(group, preset_name, cfg, dry_run=dry_run, smoke=smoke,
                           device=device)
        results.append(r)
    return results


def print_summary(results: List[dict]):
    """打印实验汇总表。"""

    print(f"\n{'='*80}")
    print("实验汇总")
    print(f"{'='*80}")
    header = f"{'实验名':<40} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} {'状态'}"
    print(header)
    print("-" * 80)
    for r in results:
        name = r.get("name", "?")[:38]
        if "error" in r:
            print(f"{name:<40} {'-':>8} {'-':>10} {'-':>8} {'-':>8} ERROR: {r['error'][:30]}")
        elif r.get("dry_run"):
            print(f"{name:<40} {'-':>8} {'-':>10} {'-':>8} {'-':>8} DRY RUN")
        else:
            val = r.get("val", {})
            status = "SMOKE" if r.get("smoke") else "OK"
            print(f"{name:<40} {val.get('map50', '-'):>8.4f} {val.get('map50_95', '-'):>10.4f} "
                  f"{val.get('precision', '-'):>8.4f} {val.get('recall', '-'):>8.4f} {status}")

    if any(r.get("smoke") for r in results):
        print("\n⚠️ 含 SMOKE 探针实验：截断训练 + 数据子采样，仅用于方向对比，不代表正式指标。")

    valid = [r for r in results if "val" in r and "error" not in r]
    if valid:
        best_map50 = max(valid, key=lambda r: r["val"]["map50"])
        best_p = max(valid, key=lambda r: r["val"]["precision"])
        best_r = max(valid, key=lambda r: r["val"]["recall"])
        print(f"\n🏆 最佳 mAP50:    {best_map50['name']} ({best_map50['val']['map50']:.4f})")
        print(f"🏆 最佳 Precision: {best_p['name']} ({best_p['val']['precision']:.4f})")
        print(f"🏆 最佳 Recall:    {best_r['name']} ({best_r['val']['recall']:.4f})")


def save_report(results: List[dict], group: str) -> Tuple[Path, Path]:
    """保存实验报告 JSON 和 Markdown（含达标检查与逐实验详情）。"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_path = REPORTS_DIR / f"optimize_{group}_{ts}.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    md_path = REPORTS_DIR / f"optimize_{group}_{ts}.md"
    lines = [
        f"# YOLO 优化实验报告 · {group}",
        f"",
        f"生成时间: {datetime.now().isoformat()}",
        f"实验数: {len(results)}",
        f"",
    ]
    if any(r.get("smoke") for r in results):
        lines.append("⚠️ 含 SMOKE 探针实验：截断训练 + 数据子采样，仅用于方向对比，不代表正式指标。")
        lines.append("")
    lines += [
        "## 汇总",
        "",
        "| 实验 | mAP50 | mAP50-95 | Precision | Recall | 备注 |",
        "|------|------:|---------:|----------:|-------:|------|",
    ]
    for r in results:
        name = r.get("name", "?")
        if "error" in r:
            lines.append(f"| {name} | - | - | - | - | ❌ {r['error'][:40]} |")
        elif r.get("dry_run"):
            lines.append(f"| {name} | - | - | - | - | dry run |")
        else:
            val = r.get("val", {})
            cfg = r.get("cfg", {})
            note = (f"model={cfg.get('model','?')} imgsz={cfg.get('imgsz','?')} "
                    f"augment={cfg.get('augment','?')}")
            if r.get("smoke"):
                note = "SMOKE 探针 · " + note
            lines.append(
                f"| {name} | {val.get('map50', '-'):.4f} | {val.get('map50_95', '-'):.4f} | "
                f"{val.get('precision', '-'):.4f} | {val.get('recall', '-'):.4f} | {note} |"
            )

    for r in results:
        if "val" not in r or "error" in r:
            continue
        name = r.get("name", "?")
        cfg = r.get("cfg", {})
        lines += [
            f"",
            f"## {name}",
            f"",
            f"- model: {cfg.get('model', '?')}",
            f"- imgsz: {cfg.get('imgsz', '?')}",
            f"- epochs: {cfg.get('epochs', '?')}",
            f"- augment: {cfg.get('augment', '?')}",
            f"- cos_lr: {cfg.get('cos_lr', False)}",
            f"- label_smoothing: {cfg.get('label_smoothing', 0.0)}",
        ]
        if r.get("smoke"):
            lines.append(f"- smoke: true（截断训练 + 数据子采样，仅用于方向对比）")
        lines += [
            f"",
            f"### val 指标",
            f"",
            f"| 类别 | Precision | Recall | mAP50 |",
            f"|------|----------:|-------:|------:|",
        ]
        for cls_name, pc in r["val"].get("per_class", {}).items():
            lines.append(f"| {cls_name} | {pc.get('precision', 0):.4f} | "
                         f"{pc.get('recall', 0):.4f} | {pc.get('map50', 0):.4f} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已保存: {json_path}")
    print(f"          : {md_path}")
    return json_path, md_path
