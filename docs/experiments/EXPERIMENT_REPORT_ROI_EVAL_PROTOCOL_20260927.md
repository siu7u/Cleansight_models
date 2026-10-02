# 实验报告：ROI 分类评估口径体检（in-sample vs 留出集）

> **状态（2026-10-02 追加）**：本报告所体检的 ROI 分类链已**评估为不采用**，配置、模型别名与
> 实验产物均已清理（实现保留）。本报告**保留为决策依据**——它正是"为什么这条链无法验收"的量化证据：
> in-sample 虚高、micro 被 91.4% 负样本掩盖、`brush_tip_out` 只有 21 个正样本致 precision 的 95% CI
> 宽达 [0.38, 0.70]。决策与理由见 [`../../MODELSET_OVERVIEW.md`](../../MODELSET_OVERVIEW.md) §9.1。

- 日期：2026-09-27
- 归属：分类（ROI）链的**评测可信度**；回答"这个数字能不能反映模型的真实能力"
- 触发：`docs/FEATURE_STRATEGY_COMPARE.md` §25.4 登记的未做项——"分类（ROI）真实数据链路：评估仍是
  in-sample（缓存由 train+val 构建），需给 test 单独建缓存"
- 性质：**纯评测口径对照，不重训**（同一个 checkpoint，只换评估数据来源）

## 0. 一屏结论

**in-sample 口径确实偏乐观，但它偏得"不均匀"——微平均几乎看不出来，逐类才现原形；而换到留出集后，
真正的问题变成"留出集本身撑不起结论"。**

| | 结论 |
|---|---|
| ① in-sample 读数不可引用 | 原报的 `f1=0.9959` / `recall=1.0` 是**在训练集上的评估**（缓存整体评估，其中约 80% 就是训练样本） |
| ② 微平均掩盖了大部分差距 | 换留出集后 micro F1 只降 **4.7pp**，exact_match 只降 **0.55pp**——因为 test 里 **91.4%** 是负样本 |
| ③ 逐类差距才是真相 | `brush_tip_out` precision 掉 **42.7pp**、F1 掉 **30.6pp** |
| ④ 但留出集也给不出可信的逐类结论 | `brush_tip_out` 在 test 只有 **21** 个正样本（in-sample 是 1028，差 **49 倍**），precision 95% CI 宽达 **[0.38, 0.70]** |
| ⑤ 能支撑的结论 | 模型显著优于两个 trivial 下界（exact_match 0.9926 vs 全零 0.9140），`air_gun`（293 正样本）F1 0.975 站得住 |
| ⑥ 不能支撑的结论 | **任何针对 `brush_tip_out` 的性能声明**——而这正是 `roi-fusion.yaml` 立项要救的淘汰类 |

## 1. 为什么要做这次体检

`roi-fusion.yaml` 的 ROI 分类链**没有 test 概念**：`build_roi_dataset` 只读 YOLO 分组数据集的
`train` 与 `val` 目录，裁剪结果缓存到 `runs/feature_fusion/datasets/<classes>/`；训练时
`_fit` 在缓存内按 80/20 随机切分，而**评估走 `predict()`，对缓存整体算指标**——也就是
**评估集合里约 80% 的样本就是训练样本本身**（不只是"同源帧近重复"，是同一批裁剪）。

原读数（2026-09-22，`runs/roi-real-airgun-brushtip/classification-20260922-113959/`）：

| precision | recall | f1 | exact_match |
|---:|---:|---:|---:|
| 0.9919 | **1.0000** | 0.9959 | 0.9981 |

`recall=1.0` + `exact_match=0.9981` 对一次 `epochs: 1` / `freeze_backbone: true` 的线性探针是
不合理的——评估系统自身也已标注该产物为 `testset.split: unknown`、`fingerprint_sha256: None`、
`validation_errors: ["配置未声明 evaluation.testset_id，无法钉定 benchmark testset"]`。

## 2. 三个口径的定义

| 口径 | 评估数据来源 | 查询方式 |
|---|---|---|
| **A. in-sample**（历史默认） | `train`+`val` 缓存整体（含训练用过的 80%） | `--config framework/experiments/roi-fusion.yaml` |
| **B. 留出集**（本次新增） | 只用 `images/test` 现算，不落缓存、不碰 A 的缓存 | `--config framework/experiments/roi-fusion-heldout.yaml` |

分组核对：YOLO 分组数据集的 `images/{train,val,test}` 在**视频/任务级完全不相交**
（train 26 组 / val 6 组 / test 43 组，两两交集为空），所以 B 是真正的跨批次留出集。

> 命名口径：图像名有两种形式（`<idx>_<8hex>-cli_<frame>` 与 `t<NN>_<frame>`），分组键分别取
> 8 位 hex 与 `task-<NN>`；无第三种形式。

## 3. 实测（同一个 checkpoint，只换口径）

checkpoint：`runs/roi-real-airgun-brushtip/classification-20260922-113959/checkpoints/best.pt`
（resnet18 / frozen / 1 epoch，SHA-256 见 `.eval.md`）

### 3.1 总体（micro 多标签）

| 指标 | A in-sample | B 留出集 test | Δ（B−A） |
|---|---:|---:|---:|
| precision | 0.9919 | 0.9194 | **−0.0725** |
| recall | 1.0000 | 0.9809 | −0.0191 |
| f1 | 0.9959 | 0.9492 | −0.0467 |
| exact_match | 0.9981 | 0.9926 | −0.0055 |

### 3.2 逐类

| 类 | 口径 | precision | recall | f1 | support |
|---|---|---:|---:|---:|---:|
| `air_gun` | A | 0.9983 | 1.0000 | 0.9992 | 3600 |
| `air_gun` | **B** | 0.9633 | 0.9863 | 0.9747 | **293** |
| `brush_tip_out` | A | 0.9698 | 1.0000 | 0.9847 | 1028 |
| `brush_tip_out` | **B** | **0.5429** | 0.9048 | **0.6786** | **21** |

### 3.3 trivial 下界（同口径，负样本占 91.4%）

留出集 3653 个裁剪 = 314 正 + 3339 负。

| 预测器 | precision | recall | f1 | exact_match |
|---|---:|---:|---:|---:|
| 全零（全判负） | —（0/0） | 0 | 0 | **0.9140** |
| 全一（全判正） | 0.0430 | 1.0000 | 0.0824 | 0 |
| **本模型（留出集）** | 0.9194 | 0.9809 | 0.9492 | **0.9926** |

模型计数（由 P/R 反解并与注册值核对）：`TP=308 / FP=27 / FN=6`。

## 4. 读数

1. **in-sample 偏乐观已确证**，但幅度**按类极不均匀**：`air_gun` 只掉 2.4pp F1，`brush_tip_out`
   掉 **30.6pp**。稀有类才是被 in-sample 口径掩盖的那一类——这与"淘汰类正是稀有类"的立项动机
   直接冲突：**用 in-sample 口径评估淘汰类，恰好会把最需要判断的类评得最好看。**
2. **micro 聚合在这份数据上不可用于判读**：test 里 91.4% 是负样本，"全判负"就能拿
   `exact_match=0.9140`；micro F1 被负样本主导，于是总体的 4.7pp 落差**低估**了真实退化
   （逐类最大落差 42.7pp）。这与时序侧 `acc` 恰好落在"全 idle 下界"上是同一个病
   （见 `docs/FEATURE_STRATEGY_COMPARE.md` §25.3 与 `tools/run_capacity_matrix.py` 的段级下界行）：
   **两个链的头条指标都在多数类地板附近失去区分力。**
3. **留出集也没有解决问题，只是把问题换了个形式**：`brush_tip_out` 在 test 只有 21 个正样本
   （train+val 是 1028，差 49 倍），其 precision 点估计 0.5429 的 95% Wilson CI 是
   **[0.382, 0.695]**、recall（19/21）CI 是 **[0.711, 0.973]**。**这个宽度不足以支撑任何
   "可用/不可用"的业务判断。**
4. **现阶段能支撑与不能支撑的结论**（这是本次体检最有用的产出）：
   - ✅ 模型确实学到了东西：`exact_match 0.9926` 显著高于全零基线 `0.9140`，`air_gun`
     （293 正样本）F1 0.975 站得住。
   - ❌ **不能**对 `brush_tip_out` 作任何性能声明；要作声明必须先补 test 侧正样本。
   - ❌ **不能**再引用 `f1=0.9959` / `recall=1.0` 这组数字（它是在训练集上算的）。
   - ⚠️ 报告里引用 ROI 分类指标必须写明口径（与 §3.4 要求写明选点口径同理）。
5. **两个口径的数字不可比，也不要"取平均"**：A 的分母含训练样本，B 的分母是跨批次留出集，
   它们回答的是两个不同问题（"拟合了吗" vs "迁移了吗"）。

## 5. 本次落地的改动

| 改动 | 位置 | 作用 |
|---|---|---|
| `data.eval_split`（可选，`train`/`val`/`test`） | `classification/pipeline.py` | 指定留出集评估：只从该 split 现算 ROI，不落缓存；留空 = 历史行为不变 |
| 配置校验 | 同上 `validate_config` | 非法 split 名在配置阶段报错 |
| `build_roi_dataset(splits=..., seed=...)` | `classification/data.py` | 可选 split 集合；负样本随机裁剪**播种**（此前未播种 → 每次重建结果不同，留出集评估不可复现） |
| `roi-fusion-heldout.yaml` | `framework/experiments/` | 留出集口径配置变体（与 `roi-fusion.yaml` 逐项一致，只多 `eval_split`） |
| 文档 | `docs/EVAL.md` §3.5、`usage/YAML_CONFIG.md` | 两个口径必须分开报；字段与配置登记 |
| 测试 | `framework/tests/test_classification_pipeline.py` | `splits` 选择语义 + 同 seed 可复现 / 换 seed 有变化 |

## 6. 复现

```bash
# 口径 A：in-sample（复现原报的 0.9959，证明默认行为未被改动）
python -m benchmark.cli.eval --config framework/experiments/roi-fusion.yaml \
    --ckpt runs/roi-real-airgun-brushtip/classification-20260922-113959/checkpoints/best.pt

# 口径 B：留出集 test（约 4 分钟，现场裁剪 3592 张图）
python -m benchmark.cli.eval --config framework/experiments/roi-fusion-heldout.yaml \
    --ckpt runs/roi-real-airgun-brushtip/classification-20260922-113959/checkpoints/best.pt

# 口径可复现性（两次独立运行必须逐项一致）
pytest framework/tests/test_classification_pipeline.py -q
```

实测产物：
`runs/roi-real-airgun-brushtip/classification-20260922-113959/evals/`
- `...-20260927-155900.evaluation.json`（口径 A，复现原读数）
- `...-20260927-155957.evaluation.json`（口径 B，第 1 次）
- `...-20260927-160212.evaluation.json`（口径 B，第 2 次，与第 1 次逐项一致 → 可复现）

## 7. 未做 / 下一步

| 项 | 状态 |
|---|---|
| **视频级切分**（train+val 内部按源视频切，而非按裁剪随机切） | 未做。缓存 `X.npy/y.npy` **不含来源信息**，要做必须在 `build_roi_dataset` 里落 provenance；本次以"留出集"这一更严格的口径替代，未引入 provenance |
| **补 test 侧正样本** | 未做，是 `brush_tip_out` 结论的唯一出路（21 → 需要至少数百个量级） |
| **正式 testset 登记** | 未做。`evaluation.mode` 仍是 `exploratory`，产物 `testset.registered=false`，要与时序链一样钉 `testset_id` 才能进正式报告 |
| **trivial 下界进常规报告** | 未做。本次为一次性计算；若要像时序侧一样常驻汇总表，需在分类评测汇总处加"全零/全一"两行 |
