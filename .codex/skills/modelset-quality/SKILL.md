---
name: modelset-quality
description: Use this skill when working on CleanSight model repositories, YOLO or temporal model benchmarks, model registry, pin.yaml, CARD.md, ModelScope packaging, or CleanSightBackend inference integration. It enforces model versioning, benchmark quality, input/output compatibility, and clean Git hygiene.
metadata:
  short-description: CleanSight modelset quality checks
---

# CleanSight Modelset Quality

## Scope

Use this skill for work under `Cleansight_models` and for model assets that will be consumed by `CleanSightBackend`.

Before changing code, benchmark scripts, configs, registry files, or docs, inspect the current repo state and preserve existing conventions.

## General Coding Discipline

Bias toward caution over speed, especially for model, benchmark, registry, and backend-integration changes.

Before implementing:

- State important assumptions when they affect the solution.
- If multiple interpretations exist, surface them instead of silently choosing.
- Ask when a missing decision would materially change the implementation.
- Push back when a simpler or safer approach fits the request better.

When implementing:

- Prefer the minimum code that solves the requested problem.
- Do not add speculative features, abstractions, configurability, or unused error handling.
- Keep edits surgical: touch only files and lines needed for the task.
- Match existing project style even if another style would also work.
- Do not refactor adjacent code, clean unrelated dead code, or reformat unrelated sections.
- Remove only unused imports, variables, or helpers introduced by the current change.

For non-trivial tasks, define success criteria and verify them:

- bug fix: reproduce or identify the failing path, then verify the fix
- validation: cover invalid inputs, then verify expected failures
- benchmark/report change: run or document the exact command and output path
- model integration: prove the intended checkpoint and mapping are actually loaded

Every changed line should trace directly to the user's request or the verification needed for it.

## Required Model Checks

For every model-related change, confirm the relevant fields:

- model type
- checkpoint path
- input shape
- `input_dim`
- `window` size
- label mapping
- feature mapping version
- dataset version
- expected inference mode

Do not claim a model is ready unless the fields needed by its deployment path are known and documented.

## Benchmark Rules

Keep benchmark levels separate:

- `benchmark/single_model`: single model quality
- `benchmark/temporal_feed_mode`: full-sequence vs streaming temporal consistency
- `benchmark/e2e_3min`: end-to-end 3-minute process evaluation

Smoke tests must be clearly marked with limits such as `--max-videos`, `--max-frames`, shortened epochs, or small sample counts. Do not describe smoke results as full benchmark results.

When reporting benchmark results, include:

- data source or split
- model checkpoint
- input feature version
- device
- metric definitions when the metric could be ambiguous
- known limits of the run

## Temporal Model Rules

Temporal model docs and benchmark reports must state:

- input shape, for example `[T, F]`, `[B, T, F]`, or streaming `[1, window, F]`
- feature dimension
- class mapping
- whether full-sequence inference is valid
- whether streaming inference is the production path
- latency measurement scope

For online deployment, prefer models with stable streaming performance, acceptable latency, and a documented relationship between offline and online features.

If a model was trained on fixed-length windows, do not assume direct full-sequence inference is valid. Compare full-sequence and streaming results before making a recommendation.

## YOLO Rules

YOLO checkpoints must document:

- Label Studio export source
- dataset version
- class list
- train/val split
- mAP or validation metric used
- per-class recall
- small-object recall when available
- checkpoint source path

A YOLO checkpoint version must not silently change its dataset, class list, or annotation source.

## Feature Mapping Rules

Feature extraction from YOLO detections to temporal input must have one canonical implementation for offline and online paths whenever possible.

When changing feature mapping, document:

- class order
- per-class channel layout
- total feature dimension
- confidence threshold
- frame sampling rate
- normalization rules
- whether the extractor is causal and stateful

Changing class order, feature dimension, thresholds, or normalization creates a new feature mapping version and usually requires retraining temporal models.

## Code Documentation Rules

Every new or substantially modified class and function must include a concise docstring or nearby comment explaining its purpose.

本仓库新增或修改代码注释、docstring 时，默认使用中文说明。已有英文标识符、CLI 参数、schema 字段、模型名和第三方术语如果会影响兼容性或可读性，应保持原样。

For classes, document:

- responsibility
- main state it owns
- expected inputs and outputs when relevant
- important model, data, or runtime assumptions

For functions, document:

- what the function does
- expected input shape or data format when relevant
- return value
- important side effects such as file writes, checkpoint loading, state mutation, or device usage

For model, feature, and benchmark code, comments must explicitly call out:

- tensor or array shape
- feature dimension
- class or label mapping assumptions
- offline vs online behavior when relevant
- whether a function is causal, stateful, or safe for streaming inference

Avoid comments that merely repeat the code, such as "increment i" or "return result". Prefer short comments that explain intent, contracts, and non-obvious constraints.

## Registry Rules

Each released model version should include:

- checkpoint file or ModelScope reference
- `CARD.md`
- `pin.yaml`
- evaluation report
- benchmark result

`pin.yaml` should pin the versions needed to reproduce the model:

- dataset
- YOLO checkpoint when relevant
- feature mapping
- temporal checkpoint when relevant
- label mapping

Changing dataset, feature mapping, class list, or checkpoint requires a new version.

## Backend Integration Rules

Before claiming a model is integrated into `CleanSightBackend`, verify:

- backend import succeeds
- configured checkpoint path exists
- model class loads
- `input_dim` matches feature mapping output
- `window` size matches checkpoint training
- label mapping matches backend output semantics
- online smoke test produces predictions
- logs prove the intended model was actually loaded

Treat the model repo as the asset and benchmark source. Treat `CleanSightBackend` as the runtime acceptance environment.

## Git Hygiene

Before preparing a commit, check status and keep generated artifacts intentional.

本仓库的 Git commit message 默认使用中文描述。若采用 Conventional Commits，允许保留
`feat`、`fix`、`refactor` 等英文 type 和英文 scope，但冒号后的提交主题必须使用中文；除非
用户明确要求其他语言。提交前应确认 message 准确概括本次实际暂存内容。

Do not commit unless explicitly intended:

- `.pt`, `.pth`, `.onnx`, `.engine`
- videos
- raw Label Studio exports with sensitive data
- `__pycache__/`
- `runs/`
- `checkpoints/`
- large generated experiment directories
- local `.env` files or tokens

Prefer committing:

- source scripts
- configs
- `CARD.md`
- `pin.yaml`
- small benchmark reports
- README or usage docs
- reproducibility notes

Preserve unrelated user changes in a dirty worktree.
