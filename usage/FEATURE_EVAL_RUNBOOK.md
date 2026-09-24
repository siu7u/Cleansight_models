# 特征方案评测 RUNBOOK（新数据集实操手册）

> **给谁看**：要在这个数据集上"新增/更换特征提取方案并出可比较指标"的队友。
> **不重复内容**：每个文件各自的字段/字段语义在 [`YAML_CONFIG.md`](./YAML_CONFIG.md)、
> 命令清单在 [`TEST_COMMANDS.md`](./TEST_COMMANDS.md)、方案索引在
> [`../docs/features/README.md`](../docs/features/README.md)、历轮结论与定版排序在
> [`../docs/FEATURE_STRATEGY_COMPARE.md`](../docs/FEATURE_STRATEGY_COMPARE.md)。
> 本文只给**执行顺序、产物落位与门禁**。
>
> 环境：一律用后端 venv —— `source /home/caizh/programming/python_code/CleanSightBackend/.venv/bin/activate`。
> 数据目录是**只读资产**：任何脚本都不得删除/重写 `datasets/` 下的数据；更新走下载器的原地增量路径。

## 1. 数据侧（只在数据集有更新时做）

```bash
# ① 有没有更新（只读、秒级）
git -C datasets/cleansight-ActionMixed-auto-lhh ls-remote origin master
git -C datasets/cleansight-ActionMixed-auto-lhh rev-parse HEAD        # 两者不一致即有更新

# ② 原地增量更新（fetch + reset --hard + lfs pull；失败不删目录）
python -m framework.cleansight_eval.cli.dataset --preset actionmixed-auto \
    --output datasets/cleansight-ActionMixed-auto-lhh
python -m framework.cleansight_eval.cli.dataset --check               # 就绪检查

# ③ 若 split 划分变了：重划 manifest（数据目录是唯一真源）
#    benchmark/manifests/actionmixed-auto/{train,val,test}.txt ← 用 labels/<split>/*.txt 的名字
#    revision = sha256(train.txt + val.txt + test.txt 原文拼接)  → 写入 framework/testsets.yaml
```

**门禁**：`python tools/validate_testsets.py --catalog framework/testsets.yaml --json` 必须
`ok=true, errors=0`。manifest 与数据目录不一致时，训练会在 `data.py` 直接抛
`FileNotFoundError: manifest 登记的动作标签不存在`（不是静默跳过）。

**LFS**：`labels/` 与 `frames/` 都是 LFS 跟踪，更新后必须有 `git lfs pull`（下载器已内置）；
否则拿到的是 pointer 文本，训练会"用到一堆 URL 字符串"。

## 2. 新增一个特征方案（清单摘录，详细见 docs/features/README.md §3）

1. `framework/cleansight_eval/temporal/features/<name>.py`：契约常量 + 逐帧构建函数（因果、无状态、中文 docstring 写明形状/语义）；
2. `temporal/data.py` 分发分支（拼接型契约在分发处组合）；
3. `framework/testsets.yaml` 登记 dataset + train/val/test 三视图（`feature_layout` / `feature_blocks` / `tail_dim` 按契约声明）；
4. `framework/experiments/<arch>-<name>.yaml`（与基线**同模型同超参**，只换特征契约）；
5. `framework/tests/test_<name>.py`（编码、空帧、遮罩、分发、维度）；
6. 文档：`docs/features/README.md` 索引 + `usage/YAML_CONFIG.md`（新增/修改任何 tracked YAML 都要）；
7. 若方案改了**模型输入结构**（如归一化 buffer），确认评估用的是 run 的 `config.resolved.json`。

## 3. 评测口径（三条铁律，缺一条结论就不可比）

1. **`--smoothing-min-duration`**：因果平滑的最小持续时长同时是召回上限。
   推荐主口径 **md=5**（历史默认 25 仅在需与旧结论对齐时用）；全序列模型无平滑，对照时应把
   因果模型也放到 md=1。
2. **逐类 support + 召回上限 + precision**：support=0 的类不可评估；只看 recall 会把
   "组级可分"误读成"方向判别"（insert/withdraw 即如此）。
3. **线性探针参照**：矩阵汇总会自动附 `linear-probe（…）` 行（train 拟合 LDA + 同 md + 同冷启动
   + 同指标实现）。方案不明显高于它，就说明换特征/加时序没有收益。

## 4. 跑矩阵（训练 + 正式评估 + 汇总）

```bash
# 全策略 × 3 seed，推荐口径 md=5
python tools/run_strategy_matrix.py --runs-dir runs/<批次名> --smoothing-min-duration 5

# 只跑部分策略 / 带超参覆盖（-S 透传）
python tools/run_strategy_matrix.py --runs-dir runs/norm_zscore \
    --strategies roi-grid-144,bbox-40-global --seeds 42,7,2026 \
    --set model.normalization=zscore --smoothing-min-duration 5

# 复用已有 run 重新汇总（幂等，秒级；--force-eval 才重跑评估）
python tools/run_strategy_matrix.py --runs-dir runs/<批次名> --skip-train --smoothing-min-duration 5
```

产物：`runs/<批次名>/<model>-<时间戳>/`（`checkpoints/best.pt`+meta、`config.resolved.json`、
`history.csv`、`evals/*.evaluation.json`、`artifacts/*.predictions.json`）、
口径变体在 `_eval_cfg/` 与 `_eval_md<N>/`，汇总在 `runs/<批次名>/STRATEGY_SUMMARY.md`。

## 5. 诊断探针（纯 CPU、只读数据、不训练；结论进文档前先跑这些）

| 工具 | 回答什么问题 | 什么时候必须跑 |
|---|---|---|
| `tools/probe_split_shift.py` | 特征有没有批次漂移？逐帧线性探针在 val/test 各能到多少？各类段长与"阈值可达性"？ | 换 test 口径、换数据版本、看到指标突变时 |
| `tools/probe_direction.py` | 某对类（默认 insert vs withdraw）可分吗？拼接上下文有帮助吗？ | 逐类 recall 长期为 0 时（先证伪"再加特征/窗口"） |
| `tools/probe_input_features.py` | 检测覆盖、动作×检测类签名、通道量纲、单变量可分性 | 新数据到位后（判断"判别物在不在"） |
| `tools/probe_pixel_channel.py` | 冻结 backbone 下整帧 vs ROI 裁剪 embedding 谁强（机制床） | 讨论像素通道是否立项时 |

## 6. 门禁清单（提交前逐条过）

```bash
python tools/validate_testsets.py --catalog framework/testsets.yaml --json     # ok=true
python -m framework.cleansight_eval.cli.dataset --check                        # 全部就绪
python -m pytest framework/tests/test_roi_features.py framework/tests/test_roi_features_v2.py \
    framework/tests/test_gru_normalization.py framework/tests/test_causal_smoothing.py \
    tests/test_team_tools.py -q                                                # 全绿
for f in $(git ls-files '*.yaml' '*.yml'); do grep -q "$(basename $f)" usage/YAML_CONFIG.md || echo "未登记: $f"; done
```

已知**既有失败**（与特征方案工作无关，别误判）：`framework/tests/test_pipeline_smoke.py::test_resume_from_last_checkpoint`
（HEAD 版 pipeline 把 `train.resume` 当路径、CLI 传布尔）、
`framework/tests/test_temporal_masking.py::test_registered_dataset_loads_only_manifest_items`
（用到未登记的 `temporal.synthetic-v1`）、`framework/tests/test_config_paths.py` 的
`actionmixed` 根目录断言（陈旧）。

## 7. 常见陷阱（都是实际踩过的）

1. **评估必须用 run 的 `config.resolved.json`**：`-S` 覆盖会改网络结构（归一化 buffer），
   用源配置评估会加载失败或按另一口径评估。
2. **口径变体要用 `apply_overrides` 生成**，别做 YAML 文本替换——resolved 配置是 JSON，
   文本替换会静默失效，出现"看着像 md=5、实际是 md=25"。
3. **eval 会把 checkpoint 报告写回 run 目录**（`checkpoints/best.eval.md`、
   `EVALUATION_REPORT.md`）。批量评估前先备份这些 md，否则 test 口径报告会被 val/变体口径覆盖。
4. **acc 会被 idle 坍缩骗**：Transformer 在本 test 上 acc 55.25 最高，但 3 seed 里 2 个
   非 idle 预测帧为 0。报 acc 必须同时给非 idle 帧数与逐类 recall。
5. **seed 方差 > 方案差异**：同策略不同 seed 的段级指标可差 6~10 个点。单 seed 不下结论。
6. **`--skip-train` 汇总扫的是目录里所有含 `config.resolved.json` 的 run**（不再只扫 `gru-*`），
   同策略同 seed 重复 run 只保留最新。

## 8. 文档落位

| 内容 | 写哪里 |
|---|---|
| 新方案语义/维度/实现/结论 | `docs/features/README.md`（索引 §0 + §1 详述） |
| 一轮实验的方法与数字 | `docs/FEATURE_STRATEGY_COMPARE.md`（按"第 N 轮"追加，结论收到"定版结论"） |
| tracked YAML 的字段语义 | `usage/YAML_CONFIG.md` |
| 命令清单 | `usage/TEST_COMMANDS.md` |
| 数据集身份/版本/revision | `framework/testsets.yaml`（+ 数据集仓库 README） |
