#!/usr/bin/env python3
"""统一模型评估入口：校验 testset 后通过 adapter 启动隔离评估。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.core.testsets import get_testset, validate_spec  # noqa: E402
from model_manager.adapters import EvaluationRequest, evaluation_command  # noqa: E402
from model_manager.catalog import ModelSpec, load_models  # noqa: E402


def request_for_spec(
    spec: ModelSpec,
    testset_id: str | None = None,
    inference_mode: str | None = None,
    device: str = "auto",
    max_videos: int | None = None,
    max_frames: int | None = None,
    append_card: bool = True,
) -> EvaluationRequest:
    """把 CLI 覆盖项与模型默认评估配置合并为固定请求。"""

    config = spec.evaluation
    resolved_testset = testset_id or config.get("testset_id")
    resolved_mode = inference_mode or config.get("inference_mode")
    if not resolved_testset or not resolved_mode:
        raise ValueError(f"{spec.id} 缺少 evaluation.testset_id/inference_mode")
    return EvaluationRequest(
        testset_id=str(resolved_testset),
        inference_mode=str(resolved_mode),
        device=device,
        max_videos=max_videos,
        max_frames=max_frames,
        append_card=append_card,
    )


def evaluate_spec(spec: ModelSpec, request: EvaluationRequest, dry_run: bool) -> int:
    """预览或执行一个模型评估；真实执行前强制通过 testset 门禁。"""

    testset = get_testset(request.testset_id)
    if testset.family != spec.family:
        raise ValueError(
            f"模型 family={spec.family} 与 testset family={testset.family} 不匹配"
        )
    command = evaluation_command(spec, request)
    print(f"[{spec.id}] evaluate")
    print(f"  testset: {testset.id}")
    print(f"  mode: {request.inference_mode}")
    print(f"  cwd: {command.cwd.relative_to(ROOT)}")
    print(f"  cmd: {' '.join(command.argv)}")
    if dry_run:
        return 0
    errors = validate_spec(testset)
    if errors:
        print("  testset validation: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 2
    return subprocess.run(command.argv, cwd=command.cwd).returncode


def parse_args() -> argparse.Namespace:
    """解析统一评估 CLI。"""

    parser = argparse.ArgumentParser(description="CleanSight 固定模型评估框架")
    parser.add_argument("--model", required=True, help="models.yaml 中的模型 id")
    parser.add_argument("--testset", help="覆盖模型默认 testset id")
    parser.add_argument("--inference-mode", help="覆盖默认推理模式")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--max-videos", type=int, help="仅 smoke test 使用")
    parser.add_argument("--max-frames", type=int, help="仅 smoke test 使用")
    parser.add_argument("--no-card", action="store_true", help="不向 CARD 追加评估记录")
    parser.add_argument("--run", action="store_true", help="真正执行；默认只预览")
    return parser.parse_args()


def main() -> int:
    """查找模型、构造请求并运行 adapter。"""

    args = parse_args()
    models = load_models()
    if args.model not in models:
        raise SystemExit(f"未知模型 id: {args.model}")
    spec = models[args.model]
    request = request_for_spec(
        spec,
        testset_id=args.testset,
        inference_mode=args.inference_mode,
        device=args.device,
        max_videos=args.max_videos,
        max_frames=args.max_frames,
        append_card=not args.no_card,
    )
    return evaluate_spec(spec, request, dry_run=not args.run)


if __name__ == "__main__":
    raise SystemExit(main())
