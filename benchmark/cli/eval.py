"""统一评测入口：python -m benchmark.cli.eval --config <yaml> --ckpt <path>。

benchmark 作为组合根按 ``framework predict → evaluator → persist/report`` 编排。
framework 只运行模型并返回 PredictionOutput；指标、artifact、EvaluationResult 和呈现均由
benchmark 定义，依赖方向保持为 benchmark → framework。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.core.artifact_io import write_json_artifact
from benchmark.core.delivery import build_delivery_manifest, write_delivery_manifest
from benchmark.core.integrity import assert_evaluation_profile, check_result_complete
from benchmark.core.provenance import (
    build_checkpoint_info,
    build_run_info,
    resolve_testset_info,
    sha256_file,
)
from benchmark.core.report import write_checkpoint_reports
from benchmark.evaluators import evaluate_prediction
from benchmark.visualizers import get_visualizer
from framework.cleansight_eval.core.checkpoint import meta_path_for
from framework.cleansight_eval.core.config import load_config
from framework.cleansight_eval.core.environment import now_stamp, pick_device
from framework.cleansight_eval.core.registry import get_pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CleanSight benchmark 统一评测入口")
    p.add_argument("--config", required=True, help="实验配置 YAML")
    p.add_argument(
        "--ckpt",
        required=True,
        help="checkpoint 路径（formal 需同名 .meta.json；exploratory 外部 YOLO 可按配置放宽）",
    )
    p.add_argument("--out-dir", default=None, help="评估结果输出目录，默认写到 ckpt 所在 run 的 evals/")
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


def _delivery_files(result, evaluation_path: Path, checkpoint_report: Path, version_report: Path, base: Path):
    """把评估事实展开为交付 manifest 输入，不复制或上传文件。"""

    checkpoint = Path(result.checkpoint).resolve()
    formal = result.run.get("evaluation_mode") == "formal"
    files = [
        ("checkpoint", checkpoint, True),
        ("checkpoint_metadata", meta_path_for(checkpoint), formal),
        ("evaluation", evaluation_path, True),
        ("checkpoint_report", checkpoint_report, False),
        ("version_report", version_report, False),
    ]
    for name, role in (
        ("config.resolved.json", "resolved_config"),
        ("env.json", "training_environment"),
        ("history.csv", "training_history"),
        ("training_curves.png", "training_curves"),
        ("status.json", "run_status"),
    ):
        files.append((role, base / name, False))
    for curve in sorted((base / "checkpoints").glob("*/results.png")):
        files.append(("training_curves", curve, False))
    for role, reference in result.artifacts.items():
        references = reference if isinstance(reference, list) else [reference]
        for item in references:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            path = Path(item["path"])
            files.append(
                (
                    f"artifact:{role}",
                    path if path.is_absolute() else base / path,
                    formal and role == "predictions",
                )
            )
    for parent in (checkpoint.parent, *checkpoint.parents):
        for name, role in (("CARD.md", "card"), ("pin.yaml", "pin")):
            candidate = parent / name
            if candidate.is_file() and not any(item[0] == role for item in files):
                files.append((role, candidate, False))
        if parent.name == "runs":
            break
    return files


def main(argv=None) -> list[str]:
    args = parse_args(argv)
    cfg = load_config(args.config)
    device = pick_device()
    pipeline = get_pipeline(cfg["pipeline"])
    pipeline.validate_config(cfg)  # 流水线专属校验（core 不再代劳）
    testset_info = resolve_testset_info(cfg)
    assert_evaluation_profile(cfg, testset_info)

    out_dir = _resolve_out_dir(args.ckpt, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction = pipeline.predict(cfg, args.ckpt, device)
    result = evaluate_prediction(prediction, cfg.get("evaluation"))
    stamp = now_stamp()
    result.timestamp = stamp
    run_info, run_dir = build_run_info(args.ckpt, args.config)
    checkpoint_info = build_checkpoint_info(args.ckpt, run_dir)
    run_info["evaluation_mode"] = cfg.get("evaluation", {}).get("mode", "formal")
    result.run = run_info
    result.testset = testset_info
    result.checkpoint_info = checkpoint_info
    result.limits = cfg.get("evaluation", {}).get("limits", {"is_smoke": False})

    artifacts_dir = (run_dir / "artifacts") if run_dir is not None else (out_dir / "artifacts")
    artifact_base = run_dir if run_dir is not None else out_dir
    for name, payload in result.pending_artifacts.items():
        artifact_path = artifacts_dir / f"{result.pipeline}-{result.model_type}-{stamp}.{name}.json"
        result.artifacts[name] = write_json_artifact(
            artifact_path,
            payload,
            relative_to=artifact_base,
        )

    # 可视化旁路直接消费本次 PredictionOutput，不重新加载 checkpoint 或重复推理。
    # 呈现失败（如缺 matplotlib）只跳过并告警，绝不拖垮正式评估事实。
    visualizer = get_visualizer(prediction.pipeline)
    if visualizer is not None and cfg.get("evaluation", {}).get("visualize", True):
        viz_dir = out_dir.parent / "viz" if out_dir.name == "evals" else out_dir / "viz"
        try:
            viz_paths = visualizer(
                prediction,
                out_dir=viz_dir,
                per_page=int(cfg.get("evaluation", {}).get("viz_per_page", 6)),
            )
            if viz_paths:
                print(f"[eval] viz: {len(viz_paths)} page(s) -> {Path(viz_paths[0]).parent}")
                result.artifacts["visualization"] = [
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

    result.integrity = check_result_complete(result)
    path = out_dir / f"{result.pipeline}-{result.model_type}-{stamp}.evaluation.json"
    result.write(path)
    print(f"[eval] {result.pipeline}: {path}")
    checkpoint_report, version_report = write_checkpoint_reports(result, path)
    print(f"[eval] checkpoint_report: {checkpoint_report}")
    print(f"[eval] version_report: {version_report}")

    delivery_base = run_dir if run_dir is not None else out_dir
    manifest = build_delivery_manifest(
        run_id=str(result.run.get("id") or stamp),
        model_id=result.model_id,
        base_dir=delivery_base,
        files=_delivery_files(result, path, checkpoint_report, version_report, artifact_base),
    )
    manifest_path = out_dir / f"{result.pipeline}-{result.model_type}-{stamp}.delivery.manifest.json"
    write_delivery_manifest(manifest_path, manifest)
    print(f"[eval] delivery_manifest: {manifest_path}")

    return [str(path), str(manifest_path)]


if __name__ == "__main__":
    main()
