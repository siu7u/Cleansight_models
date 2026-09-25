# 评审意见：《时序模型输入特征设计提案与 v3 验证计划》

- 日期：2026-09-24
- 被评审对象：一份题为《时序模型输入特征设计提案与 v3 验证计划》的提案文本（P1 scope 相对坐标系 / P2 bbox 非线性几何 / P3 priors 改辅助监督 / P4 目标 token 化 + §2 P1 验证计划 `clean_bbox_v3_scope_frame`）
- 评审性质：**静态核对 + 引用既有实测**。本评审**未跑任何新实验**，不产出新指标；结论是"能不能按这份计划往下走、需要先补什么证据"。
- 一句话结论：**方案设计本身有可取处（尤其是 P2 的 imputed 修复与 P4 的置换不变动机），但按其当前形态不能直接开工**：① 文中"已实施"在本仓库不成立；② P1+P2 的核心量在本数据集上已被实测判定"暂缓"（AUC ≈ 0.5）；③ 成功标准"至少持平"缺少配对检验与固定选点口径，**不可判读**；④ 113 维与遮罩块宽推导有确定的集成冲突。建议先做两项零/低成本前置（标注边界目检 + 可分性体检），再决定是否开工。

---

## 1. 状态核实：这份文档不在本仓库，"已实施"无法成立

| 核对项 | 结论 | 证据 |
|---|---|---|
| 是否即 `docs/features/INPUT_DESIGN_PROPOSAL.md` | **不是** | 仓库版标题为《输入设计提案：**先量可分性，再改输入**》，结构是 §0 结论先行 / §1 体检数据（1.1–1.4）/ §2 本周推荐 / §3 暂缓否掉清单 / §4 工程注意 / §5 复现命令 |
| 本提案文本是否在 `Cleansight_models` | **否** | 全仓库无可匹配的标题与章节结构 |
| 是否在相邻的 `CleanSightBackend` | **否** | 后端只有 `docs/update/20260715_OFFLINE_CLEAN_FEATURE_RECIPES.md`、`app/services/inference/offline/segmenters/clean.py`、`tests/test_offline_pipeline.py` 提到 `clean_bbox` |
| `features/clean_bbox_v3.py`（文中点名的实现） | **不存在** | `framework/cleansight_eval/temporal/features/` 下只有 `clean_bbox_v2.py` |
| `experiments/mstcn-clean-v3.yaml`（文中点名的训练配置） | **不存在** | `framework/experiments/` 下与 clean 相关的只有 `yolo-clean-{large,small}.yaml`（检测配置，非时序） |
| 对照基线配置 `experiments/mstcn-clean-v2.yaml` | **也不存在** | 同上；clean 家族当前由 `external_checkpoints/*.yaml` 驱动 |

**自洽性核对（这部分是对的）**：文中 113 维布局 `1 + 2×8 + 8×(1+8) + 7×3 + 3 = 113` ✓；且 113 / 121 / 249 与 `features.CLEAN_FEATURE_DIMS` 的三条登记（`clean_bbox_v2_top1_impute` / `+business_priors` / `+center_window+business_priors`）完全对应 ✓。

**结论**：该文本应视为**待落地计划**。"P1（第一步，已实施）"这一状态在本仓库不成立 —— 要么实现存在于其它分支/机器，要么是愿望式描述。**在补齐实现之前，第 2 节的验证计划无法执行。**

> 另一个登记口径提醒：clean 家族在 `framework/testsets.yaml` 中**未登记**（`docs/features/README.md` 记为"未登记（外部 checkpoint 配套）"）。若 v3 要进**正式轮**，必须先补 catalog 登记；若只做 exploratory，须在报告中显式声明（与 v2 同规格，不冒充正式结果）。

## 2. 核心冲突：P1+P2 的关键量已被实测判为"暂缓"

仓库版提案 §1.4 已在本数据集（12,959 帧口径）上量过"bbox 能推出的非线性量"的单变量可分性（class 3 insert, n=1784 vs class 4 withdraw, n=643）：

| 候选量 | AUC | AUC（4s 离线平滑） |
|---|---:|---:|
| hand↔scope_mid 距离 | 0.543 | 0.560 |
| hand↔scope_control 距离 | 0.445 | 0.455 |
| scope_mid 图像 y 坐标 | 0.458 | 0.456 |
| scope 长度（control↔mid） | 0.558 | 0.546 |
| hand 面积 | 0.533 | 0.519 |
| scope_mid 一阶差分 | 0.476 | 0.434 |
| **hand 在 scope 轴上的投影** | **0.469** | **0.428** |

**全部 ≈ 0.5（最好 0.558），且 4 秒离线平滑后无改善。** 其中"hand 在 scope 轴上的投影"就是 P1 的 `along`；"scope_mid 一阶差分"与 P2 的相对运动/速度同类。

在此基础上，仓库版 §3「暂缓与否掉的方向」前两条即：

| 方向 | 状态 | 依据 |
|---|---|---|
| `bbox 运动/几何非线性块（速度、Δ距离、**轴向投影**、pair 特征）` | **暂缓** | §1.4：AUC 0.43~0.56，平滑后仍 ≈0.5 |
| `参考系归一化（以 hand/scope 为原点）` | **暂缓** | 几何量本身在当前覆盖下无判别力（§1.4） |

**这两条恰好覆盖本提案的 P1（scope 相对坐标系）与 P2（pair IoU/containment + 相对运动 + 速度）。**

仓库版同时给了必要的免责与边界，本评审沿用：

> "AUC≈0.5 不等于'模型一定学不出来'（多变量组合仍可能有信息）；但它足以否掉**把该量显式加进输入**这一类提案。"

**因此，本提案要成立，必须先回答"为什么这次不一样"**，可接受的回答至少包含一项：

1. **scope 检出率显著高于做 AUC 体检时** —— P1 的轴完全依赖 `scope_control_body` / `scope_distal_end`；若检出不足，文档 §2 的回退链会退化到常数轴 `(1,0)`，等价于未做（文档成功标准 3 已识别该风险，这一点是对的，应升级为**前置门槛**而非事后检查）；
2. **坐标系变换改变了量的分布**（而非改名或等价变换）—— 需要给出变换前后分布/可分性的对照证据；
3. 或先完成零成本排除项：目检 2–3 段 insert/withdraw 视频确认**标注边界与可见动作一致**（仓库版 §1.4 的建议，约 10 分钟）。若标注本身有偏，P1/P2 的收益上限会被标注噪声压死。

## 3. 成功标准不可判读（必须改写）

文档 §2 的判据是"训练**不劣于** v2 基线（帧级 acc 与段级 edit **至少持平**）"。按本周实测证据，这不是一个可判定命题：

- `mstcn2 s4l10 h128` 逐 seed 摆幅 **0.64**，但 GRU **14.5**、Transformer **8.0**（`docs/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md` §2.3）→ seed 噪声可吞掉全部小效应；
- **同一配置只换选点口径**，test edit 即浮动 **+9.63**（p=0.0107）乃至 11 分（`usage/YAML_CONFIG.md` 的 `best_metric` 条目）→ 选点口径是隐藏变量。

建议改为可判读的四条：

1. **固定并写明 `best_metric`**（建议 `val_edit`），两臂必须同口径，写进配置而非口头约定；
2. 判据 = **逐 (seed, 视频) 配对 Wilcoxon**（工具 `tools/compare_runs.py`），报告**配对中位差 / 胜-负 / p**，不报"柱高中位数之差"（后者会被 seed 间波动稀释，见 `docs/figures/fig4_selection_metric.png` 的三数不一致现象）；
3. **强制同时报段数比与非 idle 帧数**（防"靠压掉非 idle 换 edit"的假增益）；
4. 事先声明**最小可检测效应**与 **seed 数**（本周旗舰结论用 8 seed；报告已提醒 3 seed 读数偏乐观）。

## 4. 集成风险（已定位到具体代码）

1. **113 维与遮罩块宽推导冲突**：`framework/cleansight_eval/temporal/data.py` 的 `apply_target_mask_augmentation` 在非显式块宽路径下按 `特征维 ÷ 检测类数` 推导，并做整除校验 —— 113 % 8 = 1，会**直接报错**：
   `目标随机遮罩需要特征维是检测类数 8 的整数倍，实际 113`。
   好消息是**报错而非静默错切**；但 v3 若要用 `augmentation.target_mask`，必须像 `actionmixed-roi-grid-v2` 那样在 `features.block_dims_for_version()` 登记**显式逐类块宽**（当前该函数只处理 `ROI_GRID_V2_VERSION`）。
2. **更危险的是可整除的情形**：仓库版 §4 已警告 —— 若 tail block 使总维变成 8 的倍数（例如 176 = 8×22），会**静默遮错列**（不是报错）。任何 tail block 改动都必须显式声明尾部宽度（`tail_dim` / `motion_dim` 之类字段），让遮罩只作用前缀块。这条是 `IMAGE_FEATURE_TRAINING.md` §3.5 已有的教训。
3. **登记路径**：文中给的配置落在 `experiments/`，但 clean 家族在本仓库的实际入口是 `external_checkpoints/*.yaml`；落地方需明确采用哪条路径，并同步 `usage/YAML_CONFIG.md`（受跟踪 YAML 的强制索引）。

## 5. 分项判断

| 提案 | 判断 | 依据 |
|---|---|---|
| **P2 中的 `imputed` 修复**（插值轨迹造出假速度 → 置 0 或乘 `(1-imputed)`） | **可立即独立修** | 无争议的真 bug；不依赖任何前置，也不改变接口 |
| P1（scope 相对坐标系 + 尺度/分块归一） | **条件暂缓** | §1.4 的 `along` AUC 0.469/0.428；且强依赖 scope 检出率。解锁条件：scope 检出率确认 + 可分性体检通过 |
| P2 其余（IoU/containment、相对运动、速度按尺度归一、`presence_age`） | **暂缓** | 同上；`presence_age` / `w/h` 长宽比属低风险项，可与 P1 同批体检 |
| P3（priors 从输入改为 aux head） | **建议改设计** | 动机（先验固化、捷径学习、对检测噪声脆弱）合理，但本质是"加复杂度"，而本周 13 杠杆中加复杂度类基本被判无效（加深 p=0.3776；h256+ 更差）。建议把**"直接删掉 priors"**与"改 aux head"作为**两个臂同时测**，而非只测 aux head |
| P4（目标 token 化 / set encoder） | **方向对，排最后** | 动机（置换不变、摆脱 top1/top2 槽位启发式、加类不改维度）成立，且本周结论是**换结构 > 加参数**；但同配方下 **Transformer d64×2 是最差臂（edit 27.31）**，而训练数据仅约 9.5k 帧 —— attention 类结构需要数据。且其前置是多尺度时间聚合的证据，而数据规模学习曲线协议目前**不单调、待改为随机子集** |
| 配套评估方法（独立 version + 独立报告 + 逐块消融 + permutation importance） | **采纳** | 与仓库既有纪律一致；需补 §3 的四条可判读要求 |

## 6. 建议推进顺序（成本从低到高，含止损条件）

| 序 | 动作 | 成本 | 止损/通过条件 |
|---|---|---|---|
| 1 | 目检 2–3 段 insert/withdraw，确认**标注边界**与可见动作一致；统计 **scope 类检出率** | ~10 分钟 | 标注边界系统性偏移 → 先修标注，P1/P2 全部押后；scope 检出率不足 → P1 回退链退化，P1 不成立 |
| 2 | 修 `imputed` 假速度（置 0 或 ×`(1-imputed)`） | 小 | 独立 fix，可直接提交 |
| 3 | **只做 P1 的离线可分性体检**（复用仓库版 §1.4 的探针口径：`tools/probe_input_features.py`），出 `along/across/log_area_ratio/speed` 的 AUC 表 | 低（**比训一版低两个数量级**） | AUC 仍 ≈0.5 且平滑无改善 → **止损，不进训练** |
| 4 | 通过第 3 步后，再按 §3 的可判读标准实现 v3 与对照训练 | 中 | 未达最小可检测效应 → 判为"无增益"，不进入正式轮 |
| 5 | P2 其余 → P3 → P4 排队 | 高 | 各自独立 version + 独立报告 |

## 7. 核验命令（本评审的每条事实都可复现）

```bash
PY=/home/caizh/programming/python_code/CleanSightBackend/.venv/bin/python   # 仓库根目录执行

# 状态核实：文中点名的文件是否存在
ls framework/cleansight_eval/temporal/features/clean_bbox_v3.py framework/experiments/mstcn-clean-v3.yaml

# clean 家族维度登记（应输出 113 / 121 / 249）
PYTHONPATH=. $PY -c "from framework.cleansight_eval.temporal.features import CLEAN_FEATURE_DIMS; print(CLEAN_FEATURE_DIMS)"

# 遮罩块宽推导与整除校验（113 % 8 = 1）
grep -n -A6 "block = masked.shape\[1\] // n_det_classes" framework/cleansight_eval/temporal/data.py

# 既有实测证据（§1.4 AUC 表与 §3 暂缓清单）
sed -n '86,110p;182,198p' docs/features/INPUT_DESIGN_PROPOSAL.md

# 本周证据（seed 摆幅 14.5 / 选点口径 +9.63）
sed -n '/2.3 架构族与容量/,/2.4 选点口径/p' docs/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md
```

## 8. 本评审的限制

1. **未跑新实验**：所有 AUC 与摆幅数字均引自既有文档，本评审只做静态核对与口径比对；
2. **未核实 scope 检出率**：这是第 6 节第 1 步的待办，也是 P1 成立与否的关键前置；
3. **提案来源未确认**：不排除它对应其它分支/机器上的已有实现 —— 若确实存在，应先把实现（`clean_bbox_v3.py` + 配置 + 登记 + 单测）落到本仓库，再按本评审 §3/§4 校核接口与判据；
4. **AUC 口径的固有边界**：单变量 AUC ≈0.5 只能否掉"显式加进输入"，不能否掉"多变量组合可能有效"；但本提案恰恰属于"显式加进输入"。
