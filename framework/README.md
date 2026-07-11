# cleansight_eval —— 分层训练与评估框架（骨架 + GRU 参照实现）

本目录是对 `docs/TRAIN_EVAL_REQUIREMENTS.md` 的一次落地：把"每个模型各自维护
完整训练/评估脚本"重构为一层**适度的分层框架**——共享稳定流程，但不强行统一
模型语义。当前已把 **GRU（时序）与 YOLO（检测）两条线**跑通，`task` 成为与
`family`/`execution` 对等的注册表，CLI 只做分派。

## 目录职责

| 层 | 目录 | 职责 | 与模型语义 |
|---|---|---|---|
| 框架层（骨架） | `cleansight_eval/core/` | run 组织、配置、环境、checkpoint 重建元信息、结果三态信封、评估矩阵、完整性检查 | 无关 |
| 任务层 | `cleansight_eval/tasks/` | 编排 train()/evaluate()；时序/检测的数据、预测与指标语义（含口径版本） | 同任务一致 |
| 喂入模式层 | `cleansight_eval/feeding/` | 帧怎么按时间打包给模型：窗口/因果/状态/reset（train/eval 共享，详见下节） | 按模式 |
| 特征层 | `cleansight_eval/features/` | **输入格式契约**：dim / layout / 格式版本（不含来源溯源） | 稳定契约 |
| 模型族层 | `cleansight_eval/families/` | 只放网络、forward、loss、输出转换、ckpt 兼容 | 专属 |
| 实验配置层 | `experiments/` | 族+规模+任务+喂入模式+数据+格式+训练/评估参数 | 配置 |

## 输入与喂入模式（修正后的认知）

一个模型的"输入"由两条**正交**的轴描述，谁都不吞并谁：

- **格式（feature schema）**：每个输入单元长什么样 —— `dim` + `layout`（各通道语义）
  + `version`（哪版格式）。**只讲格式，不讲来源**。时序是特征向量 schema，检测是图像
  （`modality: image, imgsz`）。
- **喂入模式（feeding mode）**：单元怎么按时间打包给模型 —— 窗口长度、因果性、
  状态/reset、读/监督哪一帧。`offline`（窗口→∞）、`realtime`（有界因果窗）、
  `single_frame`（窗口=1 无状态）都是这条轴上的**取值**。

`输入 = 格式 × 喂入模式`。换特征提取器只动"格式"，换推理协议只动"喂入模式"。

**喂入模式是 train/eval 中立的共享轴（关键，别再当成评估专属）**：

- **一个实验只有一个喂入模式，训练与评估共用它**：训练怎么喂，评估就怎么喂。不做
  "同一 checkpoint 用多种喂入分别评估"的扩展——那是多余设计。
- 训练与评估**唯一真正不同的**，是选定喂入模式之后**外面那圈机器**：训练是
  loss+反向传播，评估是算指标+出信封。这圈机器与喂入模式**正交**。
- 编排 = **Task**；建模型/前向 = **Family**；喂入模式 = **feeding**；地基（配置/设备/
  run 目录/checkpoint/信封/矩阵）= **core**。启动任一流程都需要这几者，光靠"两个协议"不够。

**已落地**：喂入模式是 `cleansight_eval/feeding/` 下与 `family`/`task` 对等的注册表
（`get_feeding`），由**顶层单个 `feeding:` 字段**表达，训练与评估共用。以 `windowed_causal`
为例：训练侧 `build_training_dataset` 造"窗口+末帧"样本，评估侧 `evaluate` 逐窗推理——同
一喂入规格的单一真源。信封字段亦从 `execution` 改名为 `feeding`。backprop 与打分外壳各自
保留（正交于喂入模式）。

## 核心不变量

- **结果三态**（`core/envelope.py`，§10）：`NOT_APPLICABLE` / `MISSING` / `COMPUTED`
  严格区分。禁止用 0 冒充 N/A、禁止缺失伪装成 N/A。
- **checkpoint 自带重建元信息**（`core/checkpoint.py`，§7.2/§8.1）：保存 family +
  模型配置 + feature schema；加载时校验，错配立即抛 `CompatibilityError`，不静默加载。
- **喂入语义显式**（§8.3）：windowed_causal 信封记录窗口、推进、冷启动、reset、平滑；
  full_sequence 绝不产生虚假实时延迟——延迟标记为 `N/A`（§8.4/§13.6）。
- **异构评估矩阵**（`core/matrix.py`，§9）：允许不同模型不同指标列，不生成综合分数。
- **不含业务门槛/自动晋升判断**（§4.5）：只产出评估事实。

## 用法（用仓库 venv）

```bash
PY=../CleanSightBackend/.venv/bin/python

# 训练（配置驱动；可用 --epochs/--lr 等临时覆盖）
cd framework && PYTHONPATH=. $PY -m cleansight_eval.cli.train --config experiments/gru-20d.yaml

# 评估（与训练同一喂入模式，出一份三态信封）
PYTHONPATH=. $PY -m cleansight_eval.cli.eval --config experiments/gru-20d.yaml --ckpt <runs/.../checkpoints/xxx.pt>

# 汇总评估矩阵（matrix.json 机读 + matrix.md 人读）
PYTHONPATH=. $PY -m cleansight_eval.cli.matrix --runs runs
```

## 测试

```bash
cd framework && ../CleanSightBackend/.venv/bin/python -m pytest tests -q
```

- `test_temporal_metrics.py`：指标口径可独立测试（§12.3）。
- `test_checkpoint_compat.py`：错配 checkpoint 拒绝加载。
- `test_envelope_matrix.py`：三态与矩阵机读/人读。
- `test_pipeline_smoke.py`：合成数据端到端 train→eval→matrix。

> 说明：真实 `Endo_Project` 数据在本机为指向 Linux 路径的软链接、不可用，冒烟测试
> 用合成数据机械验证链路。**数值对齐验收**（新 realtime 指标 == 旧
> `benchmark/temporal_feed_mode` streaming）需在有真实数据的机器上执行。

## 扩展点（后续，不在本次范围）

- 新增同架构变体：只改 `experiments/*.yaml` 的 `model` 段（§13.12）。
- 新增模型族（Transformer/causal-TCN）：加 `families/<name>/`，实现 `ModelFamily`
  协议，复用时序任务与喂入模式（§13.1/§13.2）。
- YOLO 检测任务：已实现 `tasks/detection/` + `families/yolo/` + `single_frame` 喂入模式，
  只出事实信封、纳入同一异构矩阵（§13.8）。
- stateful 喂入模式留占位（`feeding/stateful.py`）。
- `FeedingResult` 进一步泛化，让检测也走 `feeding.evaluate`（当前检测自持推理、只取 semantics）。
