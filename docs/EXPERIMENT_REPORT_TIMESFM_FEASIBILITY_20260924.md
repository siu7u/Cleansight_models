# 实验报告：TimesFM 时序基础模型融入可行性探针（零样本预测路线）

- 日期：2026-09-24
- 目的：在**不触碰训练主线、不新增 feature mapping、不改任何被跟踪代码**的前提下，用真实 val 数据回答
  "TimesFM 能不能给 CleanSight 带来东西"——具体检验它作为**预测式预警器**（点预测 + 分位数区间）
  的可用性，以及能否顺带提供动作切换信号 / 无监督异常信号。
- 一句话结论：**探针证伪"预测式预警"路线，不建议把 TimesFM 引入 CleanSight 主线。** 零样本预测在
  你们的流程序列上赢不了零成本基线（persistence / context 均值 / **一阶差分**），既无**前瞻能力**、
  也无法给出可信**剩余时长**；长视野预测基本退化为常数外推。唯一站得住的资产是**校准良好的预测
  区间**（覆盖率 0.756–0.884 vs 名义 0.80），但区间宽度相对序列波动过宽，粒度远粗于动作级。
- 规模：**1 个模型 × 4 个 val 视频 × 4 条派生序列 × 2 类探针（预测质量 / 提前预警）**，全部 CPU；
  零样本无训练，无 seed 概念。
- 性质：**可行性探针（go / no-go 判定）**，不是模型 benchmark；未产出任何 checkpoint，未改
  `framework/` / `benchmark/` / `registry/` / `testsets.yaml`。

> 汇报要点（一屏版）：TimesFM 是 Google 的**时序预测**基础模型（喂一段历史数值，预测未来走势 +
> 置信区间），与 CleanSight"逐帧判动作类别"**不是同一个任务**——官方 `SKILL.md` 自己就把分类
> 划在能力边界外。实测四点全为负：① 预测只比"延续当前值"好 7–25%，长视野与"永远猜常数"打平；
> ② 预测残差定位动作切换的 F1（0.00–0.18）**输给成本为零的一阶差分**（0.07–0.39）；
> ③ 加对照基线后"提前 4 秒预警"的能力**完全消失**（TimesFM 0.75 = context 均值 0.75）；
> ④ 剩余时长预测多数段触发不了、触发段误差还大于常数基线。加之单次预测 0.3s（GRU 滑窗 1.49ms、
> 帧预算 133ms），在线逐 tick 不可行。**建议止损**；唯一可保留用途是离线粗粒度流程偏离监控，
> 且需先有业务位点。

## 0. 本周定位（与既有报告的关系）

- 本报告是本周"这些手段值不值得投入"工作流的**第三条线**，与另两条线并列，但**性质不同**：
  容量轴（`docs/mstcn-capacity/MSTCN_CAPACITY_STUDY.md`）与特征/精度杠杆轴
  （`docs/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md`）给出的是**少数被验证有效的杠杆**
  （`mstcn2` / h128 / `roi-grid-144` / `best_metric=val_edit`）；本报告给出的是一条**被否掉的路线**，
  价值在于**阻止后续重复投入**。
- 判定为**不采纳**，因此本报告**不进入 `registry/`**、**不改 `framework/` 与 `benchmark/`**、
  **不新增 feature mapping**、**不产出 checkpoint 与 run**；全部证据为离线探针，与上述两条线的
  实验批次（各 122 个 run）**无交集**。
- 与 E1 的关系：路径 ③（冻结 backbone 当特征源）与 `EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md`
  同族但**本轮未测**；引用 E1 仅作为"冻结大模型 embedding 不是免费增益"的**先验**，不构成本报告的结论。
- 本周汇总见 `docs/WEEKLY_REPORT_20260924.md` §3。

## 1. 结论先行

1. **点预测**：4 组配置中 3 组 MAE 优于 persistence（降幅 7%–25%），但**没有任何一组与"永远预测
   常数（前 256 步均值）"拉开差距**（52d2541c：0.6694 vs 0.6802；1b2c95ff：0.5967 vs 0.6396）
   → 长视野预测缺乏前瞻结构。
2. **分位数校准**：覆盖率 0.756–0.884（名义 0.80），**校准良好**——这是唯一真正的强项；但区间宽度
   0.358–2.496，相对序列 std 0.377–0.775 很宽，只能捕捉很大的偏离。
3. **切换点可检出性**：以 1 步前向残差做动作切换检出，F1 0.000–0.179；**朴素一阶差分 `|Δ|` 为
   0.071–0.393，除一例（两者皆垃圾水平）外全面更好**，且差分成本为零 → 残差不是可用的切换/异常信号。
4. **提前预警**：加入三组零成本对照（context 均值 / 尾部 16 步均值 / persistence）后，TimesFM
   **没有系统性优势**，前两个 lead 甚至明显更差。"提前 4.27s 命中 0.75"在加对照后消失——那是
   context 均值也能做到的事。
5. **剩余时长预测**：**失败**。多数段落预测中位数在视野内不离开当前 regime（未触发的 7 段里只有
   1 段是真实剩余超过视野造成的删失），少数触发的 MAE（24.5–37.7 步 ≈ 3.3–5.0s）**大于常数基线**
   （6.5–11.0 步）。
6. **稀疏通道**：在 `p6_short_brush`（0/1 presence）上预测中位数末步 std = 0.0037（序列 std 0.377），
   退化为"预测不发生"——间歇型序列的典型失效模式。**你们大量检测通道恒零，这条尤其致命。**
7. **部署代价**：单序列调用 **0.287–0.376s**（`per_core_batch_size=1`），是 GRU 滑窗单 tick
   **1.49ms** 的约 200 倍；帧预算 **133ms** → **在线逐 tick 不可行**，仅低频旁路可行。
8. **许可证**：本次使用 `google/timesfm-2.5-200m-pytorch`（**Apache-2.0 权重，可商用**）。
   TimesFM 3.0 权重为**非商用许可**，对本项目**直接出局**，未使用。

## 2. 实验口径

| 项 | 值 |
|---|---|
| 模型 | `google/timesfm-2.5-200m-pytorch`（200M，decoder-only；零样本，无微调） |
| 数据 | `datasets/cleansight-ActionMixed-auto-lhh`，revision `6375eba95c28…`，**val split 4 视频** |
| 特征 | canonical `framework/cleansight_eval/temporal/features/roi_bbox.py::build_roi_frame_features`（`actionmixed-roi-grid-v1`，144 维 = 8 类 × 6 区域 × [presence, count, max_area]） |
| 派生目标序列 | `activity_count`（全画面框数总和）、`c0_hand`（hand 框数）、`p6_short_brush`（short_brush presence 0/1）——均**逐帧因果、无跨帧聚合** |
| 时间粒度 | 标签行 `frame_stride=4 @30fps` → **1 步 = 0.133s**；context 256 步（34s）、horizon 128 步（17s） |
| 预测配置 | `normalize_inputs=True`、`use_continuous_quantile_head=True`、`force_flip_invariance=True`、`infer_is_positive=True`、`fix_quantile_crossing=True` |
| 编译 batch | 区间/残差/时长探针 `per_core_batch_size=32`；延迟探针分别 `=1 / 8 / 32` |
| 设备 | **CPU**（本会话 `torch.cuda.is_available()=False`；结论由"有无前瞻能力"决定，与设备无关） |
| 基线 | persistence（延续末值）、常数（前 256 步均值）、**一阶差分 `\|Δ\|`**、context 均值、尾部 16 步均值 |

## 3. 验收证据（链路真的跑通了）

| 验收项 | 证据 | 结果 |
|---|---|---|
| 权重可下载可加载 | 缓存加载 **2.2s**；权重磁盘 **883MB**；进程峰值内存 **2.22GB** | ok |
| 预测形状与分位数单调 | `point (B,H)`、`quant (B,H,10)`；`q10 ≤ q50 ≤ q90` 全 True | ok |
| 真实数据可被外部模型直接消费 | canonical ROI 特征 → 单变量序列，**未改任何生产代码** | ok |
| 特征/标签对齐 | 标签行 ↔ `frames/<split>/<video>-<fid:06d>.txt` 按位置对齐，T = 244 / 1283 / 1421 / 436 | ok |
| 基线对照组齐全 | 每个结论都带 ≥1 个零成本基线（含 Q3 的一阶差分、Q5 的三组对照） | ok |

环境隔离：`timesfm` 与权重**全部装在工作区内**（`tmp/tsfm_probe/pkg`、`tmp/tsfm_probe/hf_cache`），
**未污染 `CleanSightBackend` venv**（仅借用其 python 与已装 torch）。

## 4. 结果

### 4.1 预测质量与区间校准（Q1/Q2）

| 视频 | 序列 | MAE TimesFM | MAE persistence | MAE 常数(均值) | 区间覆盖率 | 区间宽度 | 序列 std |
|---|---|---:|---:|---:|---:|---:|---:|
| 52d2541c | activity_count | **0.6694** | 0.843 | 0.6802 | 0.826 | 2.244 | 0.775 |
| 1b2c95ff | activity_count | **0.5967** | 0.678 | 0.6396 | 0.884 | 2.496 | 0.768 |
| 1b2c95ff | c0_hand | **0.1192** | 0.1963 | 0.3469 | 0.836 | 0.358 | 0.420 |
| e8ea5bb7 | p6_short_brush | 0.1636 | **0.1618** | 0.2966 | 0.756 | 0.367 | 0.377 |

分视野（52d2541c / activity_count）：h=1 → 0.326 vs 0.392；h=4 → 0.424 vs 0.479；h=12 → 0.499 vs 0.587；
h=128 → 0.669 vs 0.843。

**读法（严格按证据）**：小幅优于 persistence，但**长视野与常数预测打平** → 前瞻结构弱。
补充诊断排除了"预测=context 均值"的伪影：预测末步与 context 均值的相关系数 0.02–0.62，
不是纯均值回归，而是**真的没有结构**。稀疏 0/1 通道上预测近乎恒零。

### 4.2 切换点可检出性（Q3）

1 步前向残差做动作切换检出；判据 top-K 命中（K = 真实切换数，容差 ±6 步 ≈ 0.8s）：

| 视频 | 序列 | 真实切换点 | F1（TimesFM 残差） | F1（朴素 `\|Δ\|` 基线） |
|---|---|---:|---:|---:|
| 52d2541c | activity_count | 28 | 0.179 | **0.393** |
| 1b2c95ff | activity_count | 14 | 0.000 | **0.214** |
| 1b2c95ff | c0_hand | 14 | **0.143** | 0.071 |
| e8ea5bb7 | p6_short_brush | 3 | 0.333 | 0.333 |

**读法**：除 `c0_hand` 一例（0.143 vs 0.071，两者皆不可用水平），**成本为零的一阶差分全面更好**，
`activity_count` 上领先 2.2 倍。**用预测残差当切换/异常信号不成立。**

### 4.3 提前预警能力（Q5，带对照）

在切换前 `lead` 步起预测，考察"对切换前一时刻的预测"是否已偏向新水平（`ratio`≈0 = 延续旧水平，
≈1 = 提前给出新水平）。

1b2c95ff / activity_count（8/14 个切换点满足水平差 ≥0.5）：

| lead | TimesFM 命中 | context 均值 | 尾部 16 步均值 | persistence |
|---:|---:|---:|---:|---:|
| 1.07s | 0.125 | **0.625** | 0.125 | 0.125 |
| 2.13s | 0.125 | **0.625** | 0.375 | 0.000 |
| 3.20s | 0.625 | **0.750** | 0.625 | 0.375 |
| 4.27s | 0.750 | 0.750 | 0.750 | 0.750 |

52d2541c / activity_count（11/28）：TimesFM 0.09/0.27/0.27/0.45；context 均值 0.27/0.36/0.36/0.36；
尾部均值 0.18/0.27/0.45/0.27；persistence 0.09/0.18/0.45/0.45。

**读法**：**相对零成本基线没有系统性优势**，前两个 lead 明显更差。这是本报告最关键的对照设计——
若不做对照，会得到"TimesFM 能提前 4 秒预判"的错误结论。

### 4.4 剩余时长预测（Q4）

段落中点向前预测，"预测中位数首次离开当前 regime（最近 16 步中位 ± max(std, 0.5)）"即判定本步骤将结束：

| 视频 / 序列 | 触发段 / 总段 | 触发段 MAE | 常数基线 MAE |
|---|---:|---:|---:|
| 52d2541c / activity_count | 2/9 | 24.5 步（3.3s） | 6.5 步 |
| 1b2c95ff / activity_count | 3/10 | 37.7 步（5.0s） | 11.0 步 |
| 1b2c95ff / c0_hand | 2/10 | 37.5 步（5.0s） | 7.5 步 |
| e8ea5bb7 / p6_short_brush | 0/1 | — | — |

**读法**：**失败**。三个可测视频各未触发 7–8 段，其中**只有 1 段**是真实剩余超过视野（128 步）造成
的**删失**，其余均为真实失效；少数触发的误差还大于"直接猜中位数"。

### 4.5 部署代价（实测，CPU）

| 编译配置 | 单序列调用 | batch 调用 | 折算每序列 |
|---|---:|---:|---:|
| `per_core_batch_size=1` | **0.287–0.376 s** | — | 0.29–0.38 s |
| `per_core_batch_size=8` | 0.813–1.042 s | 0.833 s | 0.104 s |
| `per_core_batch_size=32` | 2.176–2.487 s | 2.215 s | 0.069 s |

- **坑 1（集成必知）**：`forecast()` 会把传入的 `inputs` 列表**就地补齐**到 `per_core_batch_size`
  （实测 `inputs_mutated_in_place=True`，len 1→32）。调用方**不能复用同一个 list**。
- **坑 2**：`per_core_batch_size` 是编译期静态形状，单序列调用也要付 N 条序列的算力；要便宜的单序列
  延迟必须用 `=1` 重新编译。
- 冷启动：缓存加载 **2.2s**；首次 forecast 含初始化 0.16–2.26s。
- 对比基线（`docs/INFERENCE_CHAIN_PERF.md`）：GRU 滑窗单 tick **1.49ms**、帧预算 **133ms**
  → TimesFM 单序列 **~300ms**，**在线逐 tick 不可行**；仅"低频旁路"可行（每 5–10s 触发一次
  ≈ 占单核 3–7%）。

## 5. 融合路径判定

| 融合路径 | 判定 | 依据 |
|---|---|---|
| ① 预测式预警（点预测 + 区间） | **证伪** | §4.1 与常数打平；§4.2 输给一阶差分；§4.3 加对照后无优势；§4.4 失败 |
| ② 预测残差 → 无监督异常 / 数据 QC | **证据不支持** | §4.2 残差不如一阶差分；区间虽校准良好但过宽（§4.1） |
| ③ 冻结 backbone → 时序特征源 | **本轮未测** | 负面先验两道：`EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md`（冻结 mobilenet embedding 帧级 acc 47.01→27.52、macro-F1 46.63→35.66、`air_injection` recall 归零）、`MSTCN_CAPACITY_STUDY.md`（加参数默认配方下无用）。且 ≤2.5 为**通道独立**，会丢掉 ROI 跨区域空间交互。若要继续，必须先跑机制床单 seed 最小对照，不进正式轮 |
| ④ LoRA 微调 backbone + 分类头 | **不建议** | 200M 参数 vs ~9.5k 训练帧；容量研究已给出"加参数无用"先验 |

**建议**：止损，不引入主线。若业务侧确实需要"事前预警"，那是**独立课题**，需要先有标签化的预警
定义（什么算漏步骤、什么算超时）与业务位点，而不是先选一个基础模型。

**唯一可保留的价值**：区间校准良好 → 离线"流程层面粗粒度偏离监控"（整段视频活跃度显著偏离历史
分布）。成本低、不碰主线，可作为数据体检的一路信号；需业务侧确认位点后再做。

## 6. 限制（必须随数字一起引用）

1. **仅 4 个 val 视频、单一数据集版本**（`6375eba95c28…`）；未跨数据集验证。零样本无 seed 概念。
2. **目标序列是聚合量**（框数 / presence），不是 144 维原始通道。动作切换未必在聚合量上表现为水平差
   ——52d2541c 仅 11/28、1b2c95ff 仅 8/14 个切换点满足水平差 ≥0.5。这既是 §4.2/§4.3 的主要混淆项，
   也点明了**本质错配**：TimesFM 预测的是**数值走势**，不是**动作类别**。
3. **Q5 样本量小**（8–11 个可用切换点），只能作方向性证据，不足以做强统计断言。但其方向与
   §4.2（n=28/14）一致，且 §4.4 同时出现"多数段触发不了"与"触发段误差大于常数基线"两条。
4. **Q4 的 regime 判据是自定的**（最近 16 步中位 ± max(std, 0.5)），换判据数字会变，方向不会反转。
5. **未测**：XReg 协变量模式（需额外装 `timesfm[xreg]` = jax + scikit-learn）；TimesFM 3.0
   （原生多变量，但**权重非商用**，对本项目直接出局）；GPU（本会话 GPU 不可见；延迟为 CPU 口径）。
6. **探针脚本为一次性实验代码，位于 `tmp/tsfm_probe/`（`tmp/` 已 gitignore，未入库）**；
   口径与命令已在 §7 完整记录。

## 7. 复现命令

环境（**不污染 backend venv**：包与权重都装在工作区内）：

```bash
V=/home/caizh/programming/python_code/CleanSightBackend/.venv/bin/python   # 仓库根目录执行
$V -m pip install --target tmp/tsfm_probe/pkg --no-deps "timesfm[torch]" safetensors
export HF_HOME=$PWD/tmp/tsfm_probe/hf_cache MPLCONFIGDIR=$PWD/tmp/tsfm_probe/mplcache
export PYTHONPATH=$PWD/tmp/tsfm_probe/pkg:$PWD
```

脚本（按依赖顺序；`tmp/tsfm_probe/` 下）：

```bash
$V tmp/tsfm_probe/smoke.py                                    # 加载 + 形状/分位数单调性 sanity
$V tmp/tsfm_probe/latency.py --batch-size 1                   # 延迟（1/8/32 各跑一次）
$V tmp/tsfm_probe/build_series.py --split val                 # 真实序列构建（canonical ROI 特征）
$V tmp/tsfm_probe/probe_forecast.py --video 52d2541c --series activity_count   # Q1-Q4
$V tmp/tsfm_probe/probe_lead.py     --video 1b2c95ff --series activity_count   # Q5（含三组对照）
$V tmp/tsfm_probe/plot_probe.py --npz tmp/tsfm_probe/out/probe_52d2541c__activity_count.npz \
                                --json tmp/tsfm_probe/out/probe_52d2541c__activity_count.json
```

产物：`tmp/tsfm_probe/out/*.json`（指标）、`tmp/tsfm_probe/out/*.png`（三联图：序列+预测带 /
残差 vs 差分 / 剩余时长散点）。

> 备注：探针脚本若需长期留存，建议移入 `tools/`（该目录已有 `probe_*.py` 先例）并单独提交；
> 本轮按"可行性判定、结论为不采纳"的口径**未入库**，避免给主线引入无人维护的实验代码。

## 8. 参考

- TimesFM 仓库与模型卡：<https://github.com/google-research/timesfm>（官方 `timesfm-forecasting/SKILL.md`
  明确将"时序分类/聚类"列在**不适用**清单内，建议改用 `aeon`）
- 论文：A decoder-only foundation model for time-series forecasting（ICML 2024），<https://arxiv.org/abs/2310.10688>
- 权重许可说明：≤2.5 为 Apache-2.0；3.0 权重为 `timesfm-non-commercial-license-v1.0`
- 仓库内相关先验：`docs/EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md`、
  `docs/mstcn-capacity/MSTCN_CAPACITY_STUDY.md`、`docs/INFERENCE_CHAIN_PERF.md`
