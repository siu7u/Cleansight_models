# 本周简报：2026-10-03（周六）~ 2026-10-09（周五）

> **即时更新**：每完成一项工作就追加一节，不攒到最后。
> 上级约定见 [`../README.md`](../README.md)；周索引见 [`../README.md`](../README.md) §3。

## 状态总览

| 主题 | 状态 | 结论一句话 |
|---|---|---|
| **数据集 v4.1 换代评估** | ✅ 完成 | 采样率 7.5→15fps、**标签时间轴 ×1.25（旧口径动作段整体偏早 20%）**、split 14/4/8→28/7/8 三件事同时变 → auto 链 14 个配置必须重训；检测侧逐坐标未变，YOLO 不需重训 |
| **v4.1 登记三件套** | ✅ 完成 | 新契约 `temporal.actionmixed-auto-roi-v4-v41`（revision `4aa9bf31…`）3/3 绿；revision 口径（三 manifest 原始拼接 sha256）用 3 条旧值反推验证命中 |
| **v4.1 旗舰基线重训** | ✅ 完成 | 8 seed：**acc 中位 64.91 / edit 62.64 / F1@0.1 64.40**（best.pt 经 val_edit 选点）；旧口径下恒为 0 召回的 flush / sbc / withdraw 现在召回 0.37~0.52 —— 但两把尺子不同，**差值不可直接相减** |
| **v4.1 全量重训（20 臂）+ 检测** | ✅ 完成 | 三条轴全部跑完（每臂 8 seed、正式评测）：特征轴 64.91~**76.27**、架构轴 33.75~66.37、在线轨 50.45~55.45；YOLO 复评**零回归**。**契约排名不迁移（Spearman ρ=−0.149）** |
| **会议材料 + 坏段抽查（LS 只读）** | ✅ 完成 | 一页纸（**训练中遇到的问题**，10 条：现象→证据→影响）已出；12 个整段漏检拿 Label Studio 核对 → **标注时间轴没问题**（8/8 test 视频 ×1.25 逐段命中，视频本体 30fps/468 帧独立验证）；坏段共性是**输入侧判别证据缺失**（syringe/short_brush 等逐段框数中位 0.00~0.05） |
| v3 时代旧契约 | ⚠️ 残留红 | 13 条契约 / 39 个 split 条目（12 条 auto 旧口径 + union 的 val/test 是指向 auto 的符号链接）→ 数据被覆盖，无法再绿；现报**显式**错配（不会静默读错数据） |

## 1. 一句话背景

上游数据推进到 **v4.1** 后，本地把新数据目录改名为 `datasets/cleansight-ActionMixed-auto-lhh`
——**路径与 v3 各 catalog 条目相同、内容已换代**。因此这次的核心问题是"哪些模型真的必须重训、
哪些不用"，以及"历史指标还能不能比"。

**答案**：auto 链（时序）全部必须重训且与历史不可比；YOLO 检测、legacy endo、手动 actionmixed-v2 均不受影响。

## 2. 本周产出

### 2.1 数据集换代评估（详见主题文档）

[`V41_DATA_UPDATE_20261008.md`](V41_DATA_UPDATE_20261008.md) —— 含三项实质变化的逐文件实测、
标签时间轴 ×1.25 的**独立验证**（用与标签无关的检测框数判别动作帧，AUC 0.665 vs 0.511，19/25 视频更优）、
影响面矩阵与"未能验证的部分"清单。

### 2.2 登记与工程改动

| 文件 | 改动 |
|---|---|
| `benchmark/manifests/actionmixed-auto-v41/{train,val,test}.txt` | 新增（28/7/8） |
| `framework/testsets.yaml` | 新增 `temporal.actionmixed-auto-roi-v4-v41` 数据集条目 + 3 个 split 条目 |
| `framework/experiments/mstcn2-auto-roi-v4-v41.yaml` | 新增旗舰基线配置（ref 直指 v4.1） |
| `framework/cleansight_eval/core/dataset_download.py` | `actionmixed-auto` 预设/校验路径同步到改名后的目录 |
| `usage/YAML_CONFIG.md` | 同步上述 YAML 索引 |
| `docs/MODELSET_OVERVIEW.md` | 「最新状态」块补数据换代说明 |

### 2.3 踩到的坑（值得记住）

- **不能用旧配置 + `-S data.dataset_ref=...` 切数据**：`core/config.py::load_config` 在应用 CLI 覆盖
  **之前**就做数据集门禁，旧 ref 变红后配置在**加载阶段**即失败，`-S` 来不及生效 → 必须有一份
  ref 直指新契约的配置文件。
- **目录同名换内容**是本轮最危险的形态：旧 run 的 `data.root` 与新数据**同路径**，
  所以 `--resume` 绝不能用于旧 run（`--resume` 语义错位本就已知未修）。
  数据门禁与 checkpoint 指纹是唯一防线，本轮已确认它们在按预期拦人。

## 3. 详细报告索引

| 文档 | 位置 |
|---|---|
| v4.1 换代评估（主题文档） | [`V41_DATA_UPDATE_20261008.md`](V41_DATA_UPDATE_20261008.md) |
| **周会一页纸（训练中遇到的问题）** | [`WEEKLY_ONEPAGER_20261009.md`](WEEKLY_ONEPAGER_20261009.md) |
| **坏段排查记录（Label Studio 只读核查）** | [`BAD_SEGMENTS_LS_REVIEW_20261009.md`](BAD_SEGMENTS_LS_REVIEW_20261009.md) |
| 上周（架构/配方/解码/帧准确率攻关 12 轮） | [`../2026-09-26_2026-10-02/README.md`](../2026-09-26_2026-10-02/README.md) |
| 指标口径（长期有效） | [`../../EVAL.md`](../../EVAL.md) |
| 模型集现状（长期有效） | [`../../MODELSET_OVERVIEW.md`](../../MODELSET_OVERVIEW.md) |

## 4. 遗留与下周计划

1. **v3 时代旧契约 13 条 / 39 个 split 条目**：退登记（全表回绿）／保留（整表持续 `ok:false`，
   但按 dataset_ref 的门禁不阻塞新契约）——需一次决策。其中 `actionmixed-union-roi-v1` 若继续用，
   还需按 v4.1 重建（val/test 目前是指向 auto 目录的符号链接、train 是 v3 内容副本）。
2. **`water_injection`**：train/val/test 全 0 正样本（仅隔离区 42 帧）；本轮先保留 6 类 id 重训，
   待新数据读数出来再定是否收缩为 5 类。
3. **超参重定标**：15fps 使感受野物理时长减半（`window:16` 2.13s→1.07s）；本臂刻意不动超参以保
   单变量，若基线偏弱则以此为第一候选变量。
4. **16 轮帧准确率结论的复核**：新数据同时给了正确时间轴与 +3,849 帧 lb_ins 监督，
   "insert/withdraw 方向信息不在特征里"等结论应在 v4.1 上重测。
5. **其它契约的 v4.1 版本**（40 维 / roi-v1 / roi-v5 / boxset …）：需要时按 `-roi-v4-v41` 复制。
6. **上游 README 自述不一致**（头部 revision / §二 split 表 / §数据集划分清单三处口径不一，
   只有 `task_ids.yaml` 是真值）——建议向上游提一次。
7. **val #205（`e8ea5bb7`）重叠标注**：LS 上 6 对重叠/嵌套段（详见坏段排查记录 §5）→
   请上游修正后重导出，或本地显式标记该 val 视频为缺陷样本（它参与 `val_edit` 选点）。
8. **验收口径**：是否把「逐视频 acc + 稀类段覆盖率」加进交付判据（现状 pooled acc 会掩盖 acc 0.376 的坏视频）。
9. **LS 导出前 QA**：重叠段 / 零长段 / <0.5 s 碎段检查（本项目脚本可直接复用，见 §5 工作日志 10-09）。
10. **`flush` / `sbc` 去留**：新尺子上仍是最弱两类；若下游需要，优先动检测侧（syringe / air_gun / short_brush
    召回）或换契约（`hand` / `boxset` 已在 `6d8b215c` 上显示能检出该 flush 段）。

## 5. 工作日志（即时追加）

### 2026-10-08 · 数据集 v4.1 换代评估 → **必须重训，且旧口径结论需整体复核**

- 数据本体：train 28 / val 7 / test 8，15fps（27,092 / 12,285 / 5,272 帧），`task_ids.yaml` 与目录逐 split 一致。
- **关键发现**：v4.1 标签时间轴 = v3.1 的 **×1.25**（逐边界比值中位 1.2483、段数一一对应）
  → 旧 v3.x 口径的动作段**整体偏早 20%（位置相关）**；独立佐证：检测框数判别动作帧的
  AUC 0.665（新）vs 0.511（旧），19/25 视频新口径更优。
- 检测侧逐坐标**完全相同**（最大差 0.0000）→ YOLO 不需重训。
- 影响面：auto 链 14 个配置必须重训；1,264/1,365 个历史 run 绑的数据已被覆盖 → 只读；
  legacy endo、手动 actionmixed-v2、后端均不受影响。

### 2026-10-08 · v4.1 登记三件套 → **新契约绿，旧契约残留红**

- 新增 manifest 28/7/8 + 新契约 `temporal.actionmixed-auto-roi-v4-v41`，revision
  `4aa9bf31a24789aae65ae579801e22d634e8bc3698b2dee9b25b38940fc5b600`。
- revision 口径先用 v3 / v3.1 / union 三条**已知旧值反推验证**（3/3 命中 `sha256(train‖val‖test)`）再用于新值。
- 校验：新契约 3/3 绿；`cli.dataset --check` 全就绪；残留红 = **13 条契约 / 39 个 split 条目**
  （12 条 auto 旧口径 + union），均为**显式**错配。
- 顺带修：4 个真数据测试（roi v3/v4/v5、boxset）此前因 `-lhh` 目录缺失而 skip，目录改名后开始真跑
  并按 v4.1 更新了断言的帧数（2639 → 5272）；已复跑 28 passed。

### 2026-10-09 · 可视化产物归口 + 两个可视化工具转正

**约定（用户要求，已写进被跟踪的 skill）**：所有可视化产物统一放 `outputs/visualizations/<主题>/`，
不再散落到 `tmp/`；`runs/<run>/viz/` 属该 run 的评测产物（被 delivery manifest 引用）**不得搬动**；
需版本控制的报告图仍放 `docs/figures/`。详见
[`.dsh/skills/modelset-quality/SKILL.md`](../../../.dsh/skills/modelset-quality/SKILL.md) 的「可视化产物约定」。

**同时固化了四条渲染纪律**（都踩过坑）：① 配色按类别 id **排序后的位置**取 tab10（idle 蓝 / flush 橙 /
insert 绿 / withdraw 红 / sbc 紫），**不能按类别 id 直接索引**（我第一版就把 flush 画成绿、withdraw 画成紫，
与官方图串色）；② 要对照真实视频的图**必须给秒轴**（秒 = 抽样索引 / 15）+ 上轴真实帧号，
artifact 的 `prediction_start_frame` 是 0-based 索引基准、**不是真实帧号**；③ 本机无中文字体，图中文字一律 ASCII；
④ 跨臂对比统一取「acc 中位」那个 seed 并写明口径。

**两个脚本转正为受版本控制的工具**（原先在 `tmp/`，随时会没）：

| 工具 | 作用 |
|---|---|
| `tools/plot_gt_vs_pred_bands.py` | GT vs 各臂色带图；`--axis seconds\|index`、`--arm NAME=LABEL`、`--all-arms`、`--per-page`、`--seed-pick` |
| `tools/report_band_timecodes.py` | 色带 ↔ 视频时间码：`--mode segments`（逐视频逐段明细）/ `matrix`（GT 动作段 × 各臂），`--csv` 出 Excel 表 |

产物已归口：`outputs/visualizations/bands/`（4 张图 + 4 张表）与 `outputs/visualizations/timeline/`
（roi-v5 的 8 个时间线 JSON + 2 张带状图）；被取代的旧命名文件已删除，`tmp/` 下 4 个重复脚本已清掉。

### 2026-10-08 ~ 10-09 · v4.1 全量重训（20 臂 × 8 seed）+ 检测 → ✅ 完成

**完整结果见主题文档 §7**（含逐臂表、逐类召回、溯源核对、YOLO 复评）。三条要点：

1. **特征契约排名不随数据口径迁移**：v3 旧尺子冠军 `roi-v4`/`combo`（57.67）在 v4.1 上分别
   **末位（64.91）** 与中游；旧尺子倒数第 3 的 `hand`（46.95）变成 **第 2（75.98）**。
   逐契约读数位移平均 +18.17pp，但 **Spearman ρ = −0.149（n=12）** —— 两把尺子测出了
   基本无关的优劣序。**这不是"改进 18pp"**（标签/帧数/先验都不同），是"旧排序不可用"。
2. **选点口径（val_edit）在部分臂上反噬**：`arch-transformer` 的 `best` 33.75 比 `last` 50.87
   **低 17.12pp**（唯一低于平凡基线 44.20 的臂，退化解主要来自选点而非架构）；`online-gru-roi-v1`
   低 4.77pp、`arch-actionness` 低 3.27pp。20 臂的选点虚高中位 +0.09pp、范围 −17.12~+2.02。
3. **没有全能契约**：`hand` flush 0.681 但 sbc **恒 0**；`boxset` sbc 0.783 但 flush **恒 0**；
   只有 `roi-v5` 四类召回都在 0.55 以上（唯一均衡者，edit 69.43 / macro_f1 68.84 均为最高）。
   选型必须按"要保哪个类"定，不能只看 acc。

**过程事故（如实记录）**：WSL/Windows 笔记本在 00:10~10:45 **睡眠约 10.5 小时**，批次进程被挂起后
自动续跑——**没有损坏任何产物**（已逐 seed 核 `status.json`：睡眠前那个 seed 是 `succeeded`、
60 轮完整），代价只是墙钟。另：`online-gru-*` 三臂因滑窗样本量是每臂约 1.5 小时（mstcn2 臂约 18 分钟）。

**完整性核验**：20/20 臂成功、0 失败；逐臂评测条数全为 8；`runs/v41/matrix.json` 352 行
`integrity_ok` 全 True、覆盖 12 个 v4.1 契约 testset；`SUMMARY_ALL_ARMS.md` 溯源段逐臂
`revision=4aa9bf31…` 无一例外。

**补充臂（transformer 专属 lr）也已跑完并否定原假设**：`lr=1e-4` 的选点读数 35.52
（vs 统一协议 33.75），两者都在平凡基线 44.20 之下；但两者的**固定预算**读数分别是 50.87 / 51.81
—— 说明 transformer 的"退化"主要来自**选点口径**（差 −16~−17pp），而不是架构或 lr；
其固定预算 `edit` 仅 11.5（低于全 idle 地板 13.70），段结构仍崩坏，不可交付。

**仍未做**：§6 的 5 条遗留（旧契约退登记决策、`water_injection` 5 类决策、上游 README 同步、
超参重定标、union 是否按 v4.1 重建）。

### 2026-10-08 · 全量重训批次启动 → 20 臂（三条轴）+ 修掉一个静默特征回落 bug

**契约登记扩到全部 12 个**：除 `-roi-v4-v41` 外，补齐 `auto-v41`（40 维 bbox）、`roi-v1/-v2/-v3/-v5-v41`、
`roi-presence-v1-v41`、`roi-delta-v1-v41`、`hand-v1-v41`、`global-hand-v1-v41`、`boxset-v1-v41`、
`combo-v1-v41`，共 **12 契约 / 36 个 split 条目全部绿**。条目文本由 catalog 解析器导出生成
（`tmp/v41/gen_v41_contracts.py` + `insert_v41_contracts.py`），不手抄字段。

**修掉一个真 bug（预检抓到的）**：`framework/cleansight_eval/temporal/data.py` 里
`roi_presence_recipe` 只被赋值、**分支链里没有对应 elif** → presence 契约（48 维）会静默回落到末尾
else 的 40 维 bbox 特征。`git log -S` 定位到 **2026-10-02 的 `f92fcb7`**（序列增强那次）把
`elif roi_presence_recipe:` 顶成了 `elif roi_delta_recipe:`。已恢复该分支并加防删注释；
修复后 12 个契约逐个加载维度自检 **全部一致**（presence 40→48）。

**并为它补了一条有牙的回归测试**：`test_roi_features_presence.py::test_load_split_dispatches_presence_recipe`
直接断言"加载宽度 == 契约维度"。之所以此前漏网，是因为该文件只测**特征构造器**、测不到
**加载器分派**层。已验证测试有效性：临时撤掉修复 → 该用例失败（AssertionError，宽度不符）；
恢复修复 → 7 passed。

**20 臂批次**（`tmp/v41/run_all_v41.sh`，输出 `runs/v41/<臂名>/`，8 seed / 60 轮 / `val_edit`）：

| 轴 | 臂 |
|---|---|
| A 特征轴（11） | mstcn2 s4l10 h128 × bbox40 / roi-v1 / roi-v2 / roi-v3 / roi-v5 / roi-presence / roi-delta / hand / global-hand / combo / boxset(换 `boxset_mstcn2`) |
| B 架构轴（6） | roi-grid-v4 × mstcn / transformer(→`max_len` 4096，否则 3,270 帧序列直接报错) / asformer / actionness_tcn / fact / clean_mstcn_bilstm |
| C 在线轴（3） | gru 滑窗（window 16）× bbox40 / roi-v1 / roi-v4 |

- 统一口径，唯一变量按轴变化；旗舰臂已在 `runs/v41/flagship-v41`。
- **预检（每臂 1 轮）是值得的**：19/20 直接通过，presence 那臂把上面这个静默 bug 抓了出来。

**测试三分类**（用 HEAD 的干净 worktree + 链入数据目录复现"动手前"状态做对照）：

| 类别 | 数量 | 处置 |
|---|---|---|
| 动手前就红 | 6 | 本批修掉 2 个（`test_causal_smoothing`、`test_train_video_fraction` 的 dataset_ref 换成 v4.1） |
| 数据换代导致 | 4 | 本批修掉（roi v3/v4/v5 + boxset 的帧数断言 2639→5272） |
| 剩余预存红 | 4 | `checkpoint_compat`（torch.jit 无法 script GRU 的条件 buffer）、`config_paths`（手写集根路径过期）、`pipeline_smoke::resume`（已知 `--resume` 语义错位）、`temporal_masking`（monkeypatch 未生效，测试自身问题） |

合计：**10 red → 4 red，零新增**。

### 2026-10-08 · v4.1 旗舰基线重训 → ✅ 完成（新尺子上的新参考点）

- 配置 `framework/experiments/mstcn2-auto-roi-v4-v41.yaml`；启动脚本 `tmp/v41/run_flagship_v41.sh`；
  输出 `runs/v41/flagship-v41/`（8 seed，60 轮，`best_metric=val_edit`，用时 16:54:46~17:10:16）。
- 与 `runs/acc-push/acc-roiv4/h128` 逐项同配方 → 数据版本是本臂**唯一变量**。
- **溯源已核**：每条 evaluation.json 记 `dataset_version=cleansight-actionmixed-auto-roi-v4-v41`、
  `dataset_revision=4aa9bf31…`、`num_items=8`、`validation_errors=[]`。

**test（best.pt，8 seed 中位）**

| 指标 | v4.1（新尺子） | v3 旧口径 | 说明 |
|---|---:|---:|---|
| acc | **64.91**（62.18~67.34） | 57.59 | 见下方"不可相减"说明 |
| edit | **62.64**（54.49~68.77） | 42.90 | 同上 |
| F1@0.1 / 0.25 / 0.5 | **64.40 / 56.57 / 36.30** | 39.83 / 31.89 / 13.03 | 同上 |
| frame macro F1 | **56.96** | 26.80 | 同上 |

**逐类帧召回（8 seed 中位）** —— 本轮最值得注意的一列：

| 类 | v4.1 支持帧 | 召回 v4.1 | 召回 v3 | 精确 v4.1 | 精确 v3 |
|---|---:|---:|---:|---:|---:|
| idle | 2,330 | 0.869 | 0.886 | 0.635 | 0.607 |
| flush | 408 | **0.368** | **0.000** | 0.731 | 0.000 |
| long_brush_insert | 1,765 | **0.510** | 0.299 | 0.740 | 0.617 |
| long_brush_withdraw | 642 | **0.393** | 0.069 | 0.527 | 0.224 |
| short_brush_cleaning | 127 | **0.524** | **0.000** | 0.731 | 0.000 |
| water_injection | 0 | n/a | n/a | n/a | 0.000 |

**为什么不能把上表右列直接相减**（两条独立理由）：

1. **标签换了**：v4.1 是 ×1.25 修正后的时间轴，旧列测的是模型在**错位标签**下的表现
   —— 两列衡量的不是同一件事。
2. **先验换了**：新 test 的 idle 占比 44.20%，旧 test 是 55.25% → **平凡基线（全预测 idle）
   本身就降了 11.05pp**，acc 天然上移。相对平凡基线：新 +20.7pp、旧 +2.3pp。

**可下的结论**：旧口径下**恒为 0 召回**的 flush / short_brush_cleaning 与近乎为 0 的 withdraw
（0.069），在新数据同配方下召回 0.37~0.52、精确 0.53~0.73 —— **与此前 16 轮攻关反复归因的
"模型/特征/损失都不行"相反，这批失效模式更像由监督侧（错位标签 + 仅 14 个训练视频）造成**。
这是**方向性证据**，不是幅度结论：要谈幅度必须在 v4.1 上重跑对照臂（v1/v2/v5/boxset 等）。

### 2026-10-09 · 会议材料（训练中遇到的问题）+ 坏段 Label Studio 只读抽查 → ✅ 完成

**产出**：[`WEEKLY_ONEPAGER_20261009.md`](WEEKLY_ONEPAGER_20261009.md)（5 分钟一页纸）、
[`BAD_SEGMENTS_LS_REVIEW_20261009.md`](BAD_SEGMENTS_LS_REVIEW_20261009.md)（坏段排查记录）。

**一页纸口径（10-09 晚改版：改为问题导向）**：主线从"试过的方向 × 读数 × 判定"换成
**"训练中遇到的问题（现象 → 证据 → 影响 → 待拍板）"**，共 10 条，分三层：口径类（P1 契约排名不迁移
ρ=−0.149 / P2 `val_edit` 选点反噬 −17.12pp / P9 指标掩盖坏结果）、数据与标注类（P3 稀类无输入证据 /
P4 整段漏与方向互换 / P5 同段别的臂能检出 / P6 val #205 重叠标注）、工程与流程类（P7 静默特征宽度回落 /
P8 目录同名换内容 / P10 睡眠挂起 10.5 h）。
数字仍分两类并**禁止相减**——(A) 本周 v4.1 尺子实跑的三轴 22 臂（`runs/v41/`，8 seed 中位、best.pt）；
(B) 此前 v3 尺子的历史读数（引自既有报告原文，本次未复算）。
原"方向 × 评测数据"四张表压缩为**附录 A 读数速查**（特征轴 12 契约 64.91~76.27 / 架构轴 8 臂 33.75~66.37 /
在线轴 3 臂 50.45~58.82 / 检测侧零回归），三条硬结论已并入 P1~P3 的证据列。

**坏段排查（Label Studio 只读，未写入任何标注）**：

- 选段口径：旗舰臂 8 条 best.pt 评测中取 acc 最接近中位的 seed（`mstcn2-20261008-170149`，acc 64.38）。
  逐视频 acc **0.376~0.779**；**整段漏检 12 段**（动作段 10 + idle 段 2）、误报段 18 个、部分命中段 15 个。
- **LS 侧核对**：project 18（test 8 视频）LS 原始标注与本地标签**逐段一一对应**，本地帧号 = LS 帧号 × **1.25**
  （比值 ≈1.000，8/8 视频）；并用 **LS 取回的原始视频**独立验证（`ffprobe`：30/1 fps、468 帧 / 15.60 s ↔ 本地末帧 467）
  → **×1.25 口径成立，坏段不能归因到标签错位**。3 处比值 2.4 已逐条确认为"LS 首段起点 = 帧 1"的取样量化，非缺陷。
- **失败三模式**：① 稀类短段全漏（`6f1a85e3` 的 sbc **5/5 coverage=0**）；② insert↔withdraw 方向互换
  （`ac38ca6b` 2 段整段漏 + 9 段误报）；③ flush 整段漏（`6d8b215c` 9.67 s 全判 idle）。
- **输入侧证据（35 个 test 动作段）**：`syringe`/`air_gun`/`short_brush`/`brush_tip_out` 逐段框数中位
  **0.00~0.05**（8 段 sbc 全为 0.00）；`67edf4f9` 逐 0.5 s 轨迹显示 idle 段与 flush 段的检测组成**完全相同**
  （hand 2 / scope_control 1 / scope_mid 1）→ 稀类在特征层没有直接证据。
- **同段有臂能检出**：`6d8b215c` 上旗舰"全 idle"，而 `feat-hand` 预测 flush 4.10–14.63 s（acc 0.795）、
  `feat-boxset` acc 0.816 → 证据在输入里，`roi-v4` 的通道选择没送到模型面前（**相关性，非因果**，单 seed）。
- **LS 标注侧缺陷（不在 test）**：project 16 共 **7 对重叠段**——val 视频 #205 占 6 对（13.8 s 的 sbc 长段内嵌 4 段
  + flush 两对交叠，本地导出取并集：8 flush→6、6 sbc→2）、train #197 1 对（含 0.00 s 零长段）。
  **#205 是 val 成员而选点用 `val_edit`** → 会污染选点。test 侧 project 18：重叠 0 / 零长 0 / <0.5 s 碎段 0。
- **sbc 碎片化**：LS 原始标注就把 sbc 切成 0.5~1.7 s 小段、段间隔仅 0.08~0.3 s（`6f1a85e3` 5 段、`c7c853a4` 3 段）
  → 段级指标对各碎片分别记账，是 M1 的一部分成因。

**复现**（**已入库**：`tools/ls_segment_review.py`，只读，`compare` = LS↔本地逐段对拍 + 标注质量扫描、
`failures` = 逐视频/逐段失败画像；临时脚本另在 `tmp/ls_check/`，未入库）：
`compare_ls_local.py`（LS↔本地逐段对拍）、`arm_video_matrix.py`（22 臂 × 8 视频）、`det_evidence.py`（检测证据）；
色带图 `outputs/visualizations/bands/gt_vs_pred_tc_p01.png`（`tools/plot_gt_vs_pred_bands.py`，含逐视频 acc）。
关键帧截图与 LS 取回的原始视频留在 `tmp/ls_check/{frames,videos}/`（scratch，可删）。
导出 CSV：`outputs/ls_review/{ls_vs_local_test,ls_annotation_qa_test,seg_failures_flagship-v41}.csv`（入库工具的产物）
与 `tmp/ls_check/{ls_vs_local_segments,arm_video_matrix,seg_detection_evidence,seg_failures_flagship-v41}.csv`（临时脚本产物）。

**遗留（登记进 §4）**：val #205 重叠标注处置；验收口径是否加"逐视频 acc + 稀类段覆盖率"；
LS 导出前加"重叠/零长/碎段"QA；`flush`/`sbc` 去留。

### 2026-10-09 · 叠加预测视频（"视频进程里显示当前预测动作"）→ ✅ 完成

**产出**：`outputs/visualizations/pred_v41/`（旗舰臂中位 seed，8 个 test 视频各一条 `<序列>_pred.mp4`）
与 `outputs/visualizations/pred_v41_cmp/`（同一视频 `6d8b215c` 的 `feat-hand` / `feat-roi-v5` 对照条），
九宫格预览 `outputs/visualizations/pred_v41/preview_6d8b215c_3arms.png`、`preview_67edf4f9_1arm.png`。

- **工具**：沿用既有 `tools/visualize_predictions.py`（只读已存盘预测产物，**不加载 checkpoint、不重新推理**）：
  顶部横幅 = 当前抽样帧的预测动作（Pred，动作色填充），其下小字条 = 人工真值（GT，不符时标 `MISMATCH`），
  画面内叠加 YOLO 检测框。
- **像素源**：本数据集**不含像素**，原始视频只在 Label Studio 上 → 从 LS 取回 8 个 test 视频到
  `outputs/visualizations/pred_v41/_src/`（`/data/upload/18/<file>`，只读；脚本 `tmp/ls_check/lsclient.py`）。
- **输出时间轴**：`--fps 15`（采样步长 2、源视频 30 fps → 15 fps 即**实时播放**），逐视频 frame-acc
  与坏段排查记录的逐视频 acc 完全一致（37.6 / 39.9 / 57.7 / 71.4 / 73.5 / 76.6 / 76.6 / 77.9）。
- **顺带修了工具的两处口径**（改动仅 `tools/visualize_predictions.py`）：
  ① 横幅配色改为**按"本产物里出现过的类别"排序取 tab10 前 n 色**（与 `benchmark/visualizers/temporal.py`
  色带图同色：idle 蓝 / flush 橙 / insert 绿 / withdraw 红 / sbc 紫；此前按动作 id 直接索引会得到
  flush 红、idle 绿的错配 —— 正是 skill 里警示过的那种错配）；② 横幅右侧补真实时间码
  （`frame 271  t=9.0s`），便于与报告里的秒级时间码对照。
- **复现**：
  ```bash
  python tools/visualize_predictions.py \
    --artifact runs/v41/flagship-v41/h128/mstcn2-20261008-170149/artifacts/full_sequence_temporal-mstcn2-20261008-170942.predictions.json \
    --dataset datasets/cleansight-ActionMixed-auto-lhh \
    --video outputs/visualizations/pred_v41/_src/<序列>.mp4 --sequence <序列>.mp4 \
    --split test --fps 15 --out-dir outputs/visualizations/pred_v41
  ```
- **注意**：`outputs/` 不入库（可再生），`_src/` 里的源视频与 `_check/` 的抽帧只作现场查看用。
