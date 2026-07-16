"""评估入口：python -m cleansight_eval.cli.eval --config <yaml> --ckpt <path>。

只做**分派**：按 ``cfg["pipeline"]`` 调用 ``get_pipeline(...).evaluate(...)``，得到一份三态
信封并落盘。重建模型、指标口径、推理语义等由所属流水线实现。训练与评估同属一条流水线，
输入构造与输出语义一致，不做多模式扫描。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core.artifacts import write_json_artifact
from ..core.config import load_config
from ..core.environment import now_stamp, pick_device
from ..core.integrity import check_envelope_complete
from ..core.provenance import build_checkpoint_info, build_run_info, resolve_testset_info, sha256_file
from ..core.report import write_checkpoint_reports
from ._registry import get_pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="cleansight_eval 评估入口")
    p.add_argument("--config", required=True, help="实验配置 YAML")
    p.add_argument("--ckpt", required=True, help="checkpoint 路径（需存在同名 .meta.json）")
    p.add_argument("--out-dir", default=None, help="信封输出目录，默认写到 ckpt 所在 run 的 evals/")
    return p.parse_args(argv)


def _resolve_out_dir(ckpt: str, override: str | None) -> Path:
    if override:
        return Path(override)
    # ckpt 通常在 <run>/checkpoints/... 下，评估写到同 run 的 evals/。
    # 时序权重直接在 checkpoints/；检测权重在 checkpoints/<name>/weights/，故向上找。
    ckpt_path = Path(ckpt)
    for anc in ckpt_path.parents:
        if anc.name == "checkpoints":
            return anc.parent / "evals"
    return ckpt_path.parent / "evals"


def main(argv=None) -> list[str]:
    args = parse_args(argv)
    cfg = load_config(args.config)
    device = pick_device()
    pipeline = get_pipeline(cfg["pipeline"])
    pipeline.validate_config(cfg)  # 流水线专属校验（core 不再代劳）

    out_dir = _resolve_out_dir(args.ckpt, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    envelope = pipeline.evaluate(cfg, args.ckpt, device)
    stamp = envelope.timestamp or now_stamp()
    run_info, run_dir = build_run_info(args.ckpt, args.config, device)
    checkpoint_info = build_checkpoint_info(args.ckpt, run_dir)
    envelope.run = run_info
    envelope.testset = resolve_testset_info(cfg)
    envelope.checkpoint_info = checkpoint_info
    envelope.limits = cfg.get("evaluation", {}).get("limits", {"is_smoke": False})

    artifacts_dir = (run_dir / "artifacts") if run_dir is not None else (out_dir / "artifacts")
    artifact_base = run_dir if run_dir is not None else out_dir
    for name, payload in envelope.pending_artifacts.items():
        artifact_path = artifacts_dir / f"{envelope.pipeline}-{envelope.model_type}-{stamp}.{name}.json"
        envelope.artifacts[name] = write_json_artifact(
            artifact_path,
            payload,
            relative_to=artifact_base,
        )

    # 可视化旁路（duck-type 钩子，仅个别流水线提供）：出图便于评估时肉眼快速发现错分。
    # 隔离于信封主流程之外——出图失败（如缺 matplotlib）只跳过并告警，绝不拖垮评估。
    if hasattr(pipeline, "visualize"):
        viz_dir = out_dir.parent / "viz" if out_dir.name == "evals" else out_dir / "viz"
        try:
            viz_paths = pipeline.visualize(cfg, args.ckpt, device, viz_dir)
            if viz_paths:
                print(f"[eval] viz: {len(viz_paths)} page(s) -> {Path(viz_paths[0]).parent}")
                envelope.artifacts["visualization"] = [
                    {
                        "path": str(Path(viz_path).resolve().relative_to(artifact_base.resolve()))
                        if Path(viz_path).resolve().is_relative_to(artifact_base.resolve())
                        else str(Path(viz_path).resolve()),
                        "sha256": sha256_file(viz_path),
                    }
                    for viz_path in viz_paths
                ]
        except Exception as exc:  # 呈现层不影响评估事实的产出
            print(f"[eval] viz skipped: {exc}")

    envelope.integrity = check_envelope_complete(envelope)
    path = out_dir / f"{envelope.pipeline}-{envelope.model_type}-{stamp}.envelope.json"
    envelope.write(path)
    print(f"[eval] {envelope.pipeline}: {path}")
    checkpoint_report, version_report = write_checkpoint_reports(envelope, path)
    print(f"[eval] checkpoint_report: {checkpoint_report}")
    print(f"[eval] version_report: {version_report}")

    return [str(path)]


if __name__ == "__main__":
    main()
