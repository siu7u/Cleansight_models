"""检测任务实现（DetectionTask）。

消费 cleansight-yolo-pipeline 产出的标准 YOLO 数据集（images/labels/data.yaml），
用 ultralytics 训练/验证，**只产事实信封**：mAP / P / R 逐类三态指标 + 完整性，
不含任何业务门槛、不判 PASS/FAIL、不设非零退出码（对齐 §4.5 / §10 / §13.11，
替代 pipeline 里带门禁的 04_validate.py）。

喂入模式用 ``single_frame``：单帧无状态。检测不经由 ``feeding.evaluate``
（那套返回时序逐帧数组），只取其 ``semantics`` 挂进信封说明喂入语义。
"""

from __future__ import annotations

import json

from ...core.checkpoint import load_meta, meta_path_for
from ...core.environment import now_stamp, set_seed
from ...core.envelope import EvalEnvelope
from ...core.integrity import assert_checkpoint_config, check_envelope_complete
from ...feeding import get_feeding
from ...feeding.perf import not_applicable_perf
from ...families import get_family
from ...core.run import RunContext
from .metrics import build_detection_metrics


class DetectionTask:
    task_id = "detection"

    def validate_config(self, cfg: dict) -> None:
        data = cfg.get("data", {})
        if "data_yaml" not in data:
            raise ValueError("检测任务 data 段需包含 data_yaml（指向 YOLO 数据集的 data.yaml）")
        get_feeding(cfg["feeding"])  # 喂入模式须已注册（检测通常为 single_frame）

    def train(self, cfg: dict, runs_dir: str, seed: int, device) -> str:
        set_seed(seed)
        family = get_family(cfg["family"])
        model_cfg = cfg["model"]
        train_cfg = cfg.get("train", {})

        run = RunContext(runs_dir, family=cfg["family"])
        run.save_config(cfg)
        run.save_env(device, seed=seed)

        data_yaml = cfg["data"]["data_yaml"]
        name = cfg["data"].get("name", "detect")
        best_pt, num_params, names, nc = family.train(
            weights=model_cfg.get("weights", "yolo11n.pt"),
            data_yaml=data_yaml,
            train_cfg=train_cfg,
            imgsz=model_cfg.get("imgsz", 640),
            device=device,
            project=str(run.checkpoints_dir),
            name=name,
        )

        # sidecar 重建/溯源元信息：让 YOLO 权重也能进异构矩阵、被 load_meta 校验（§7.2/§8.1）。
        meta = {
            "family": cfg["family"],
            "task": cfg["task"],
            "nc": nc,
            "names": names,
            "num_params": num_params,
            "dataset": name,
            "data_yaml": str(data_yaml),
            "model": model_cfg,
            "train": train_cfg,
            "trained_at": now_stamp(),
        }
        meta_path_for(best_pt).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[train] run_dir={run.dir}")
        print(f"[train] checkpoint={best_pt}")
        return str(best_pt)

    def evaluate(self, cfg: dict, ckpt: str, feeding_name: str, device) -> EvalEnvelope:
        meta = load_meta(ckpt)  # 缺 sidecar 直接报错，拒绝盲加载（§8.1）
        assert_checkpoint_config(meta, {"family": cfg["family"]})

        family = get_family(cfg["family"])
        mode = get_feeding(feeding_name)  # single_frame：只取其 semantics

        val = family.val(
            weights=ckpt,
            data_yaml=cfg["data"]["data_yaml"],
            split=cfg["data"].get("eval_split", "val"),
            imgsz=cfg["model"].get("imgsz", 640),
            device=device,
        )
        metrics = build_detection_metrics(val)
        performance = not_applicable_perf(reason="单帧检测评估不测实时延迟")

        envelope = EvalEnvelope(
            family=cfg["family"],
            model_id=f"{cfg['family']}-{cfg['data'].get('name', '')}",
            task=cfg["task"],
            feeding=feeding_name,
            checkpoint=str(ckpt),
            dataset=cfg["data"].get("name", cfg["data"]["data_yaml"]),
            feature_schema={"modality": "image", "imgsz": cfg["model"].get("imgsz", 640)},
            metrics=metrics,
            performance=performance,
            feeding_semantics=getattr(mode, "semantics", {}),
            num_params=meta.get("num_params"),
            timestamp=now_stamp(),
        )
        envelope.integrity = check_envelope_complete(envelope)
        return envelope
