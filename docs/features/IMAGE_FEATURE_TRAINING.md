# 图像特征训练流程（IMAGE_FEATURE_TRAINING）

> **状态（2026-09-03）**：**正式训练方案已定稿**（§3.4：ROI 网格 144 + 健康配方，由多 seed
> 对照证据确定）；图像通道（形态 B）降级为**候选增强实验**（§4 E 系列）——只在消融证明有
> 段级增益后并入正式方案。本文档是训练流程总纲：决策记录 → 现状与证据 → 正式方案 →
> 候选增强实验 → 前置阻塞 → 执行清单。
>
> 相关文档：[特征提取方案索引](README.md)（已实现契约）、
> [`FEATURE_STRATEGY_COMPARE.md`](../FEATURE_STRATEGY_COMPARE.md)（bbox 系对照结论与坍缩分析）。

## 1. 目标与决策记录（2026-09 团队讨论结论）

| 决策 | 内容 | 落点 |
|---|---|---|
| D1 | **图像信息加入 bbox 特征**（多模态融合），不做"纯图像"时序模型（纯图像受画面干扰因素影响大、难训出） | 本文档第 4 节实验矩阵 E1/E2 |
| D2 | 图像特征用**现成预训练 backbone**（先冻结权重），**不训练第二套视觉模型**（与 YOLO 功能重叠，且小数据训不动） | 冻结 + 轻量投影头（§4.3） |
| D3 | 前置 CNN **零训练**——全链路只调时序（+投影）一套参数，不调两套 | §4.3 |
| D4 | 开 action-test 采集**长短毛刷刷洗**测试数据（补 v3 test 缺失的 sb_clean 覆盖） | §5.2 |
| D5 | 本周交付物 = 实验确定的图像特征提取方案（含对照证据） | 本文档 §6 验收 |
| D6 | **正式训练方案以多 seed 实测为准**：bbox 系主线定为 ROI 网格 144 维（唯一三 seed 全不坍缩、中位段级指标领先，见 FEATURE_STRATEGY_COMPARE.md 第三轮矩阵） | 本文档 §3.4 |
| D7 | **形态 A（独立 ROI 图像分类）移出训练主线**：模型内无位置/时序信息、与 bbox 位置通道职责重叠、分类器从未训练——训练文档不再展开，检测侧（淘汰类）作为独立方向保留参考 | 本文档 §2 注 |

## 2. 训练文档收口：唯一在案的图像方案是"像素特征进时序"

本文档只保留**一种**进入时序训练的图像方案：

| 方案 | 定义 | 现状（2026-09-03） |
|---|---|---|
| **像素特征进时序（形态 B）** | 每帧 CNN embedding 作为时序输入 `[B,T,F]` 的像素派生通道（冻结预训练 backbone，不训练第二套视觉模型） | **提取工具已落地**（§3.1），训练侧接入未实现（§4 为候选增强实验） |

> 注（D7）：早前的"独立 ROI 图像分类"（`roi_classification`/`feature_fusion`，对裁剪块做多标签分类）
> 已**移出本文档训练主线**——该形态的模型输入只有裁剪图，帧内位置与时间关系均不进模型，
> 与 bbox 位置通道职责重叠；分类器也从未正式训练。检测侧的淘汰类替代（P/R<0.3 的类）属
> 独立方向，见 `docs/YOLO_OPTIMIZATION.md` §4，不在本文档展开。

## 3. 现状盘点：已就绪的积木

### 3.1 已落地：整帧 embedding 离线预计算工具

`framework/cleansight_eval/temporal/features/extract_embeddings.py`（提交 7ec3b99）：

```bash
python -m framework.cleansight_eval.temporal.features.extract_embeddings \
  --root <数据集> --splits train,val,test --backbone resnet18 --out-dir <产物根>
```

- 产物：`<out>/<split>/<video>.mp4.npy`（`[T, feat_dim]`，与标签行一一对齐）+ `meta.json`
  （backbone/输入尺寸/ImageNet 预处理/缺图记录）
- 语义：因果（只看当前帧）、确定性（eval+no_grad）、缺图帧补零不静默错位（有单测保护）
- backbone：resnet18/34/50、mobilenet_v3_small、efficientnet_b0（与 classification 同权重口径）
- 工具链路已通过**全量对齐机制验证**（2026-09-03：9,532 帧图文严格对齐、缺图 0、
  CPU 提取约 40s，单测保护在案）
- **正式产物以最新数据集（`datasets/cleansight-ActionMixed-auto-lhh`）为目标**：该数据集
  按设计不含像素（见 §5.1 核心阻塞），待像素源（LS project-16 下载抽帧 / action-test 采集
  保留帧图）就绪后，在 -lhh 视频上重跑全量提取、对齐校验并登记为新数据契约；此前的机制
  验证产物仅留作链路调试，不进入该数据集契约
- backbone 权重就位：resnet18/34/50 + mobilenet_v3_small + efficientnet_b0 已入
  torch 默认缓存（`~/.cache/torch/hub/checkpoints/`），离线可用

### 3.2 已移植：队友图像管线参考（tools/，提交 7ec3b99）

| 工具 | 用途 | 可复用件 |
|---|---|---|
| `tools/compare_roi_backbones_for_tracking.py` | ROI backbone 对比（ReID） | `imread_unicode` / `crop_detection`（框裁剪+padding+resize）/ `build_backbone`（5 种）/ batch 推理 / fp16 / 计时 |
| `tools/benchmark_gpu_roi_track_latency.py` | GPU 吞吐/显存实测 | 实测参考：ResNet18 1092 ROI/s、MobileNet 133 MiB（RTX 4060） |
| `tools/evaluate_labelstudio_trackers.py` 等 | LS 帧图/GT 加载 | 帧路径解析 |

> 注意：队友工具面向 **tracker ReID**（per-检测框 embedding），不是时序动作输入——**中间件可复用，任务语义需自建**。

### 3.3 已就绪：bbox 系主线与多 seed 对照证据

- bbox 系四策略（40/40 手/80 双通道/144 ROI 网格）健康配方代码已落地（dropout/best 指标可选/
  早停/权重截断，提交 0059eb9），一键多 seed 矩阵工具 `tools/run_strategy_matrix.py`（43f15ef）
- **多 seed 矩阵（42/7/2026，test 锚定 task#195/#199）**：

| 策略 | 中位 F1@0.1 | 中位 F1@0.25 | 坍缩 seed 数 |
|---|---:|---:|---:|
| **ROI 网格 144** | **31.8** | **22.7** | 0/3 |
| 全局+手部 80 | 15.8 | 10.5 | 1/3 |
| 手部 40 | 11.8 | 5.9 | 2/3 |
| 全局 40 | 11.8 | 5.9 | 2/3 |

GPU 口径（2026-09-04，RTX 4060 Laptop，数据根 `-lhh`，同配方同 seed；坍缩 = 非 idle 预测帧 0）：

| 策略 | 中位 F1@0.1 | 中位 F1@0.25 | 坍缩 seed 数 |
|---|---:|---:|---:|
| **ROI 网格 144** | **19.5** | **14.6** | 0/3 |
| 全局 40 | 20.0 | 15.0 | 0/3 |
| 全局+手部 80 | 11.8 | 5.9 | 1/3 |
| 手部 40 | 11.8 | 5.9 | 3/3 |

（完整逐 seed 表见 FEATURE_STRATEGY_COMPARE.md 第三/四轮；"坍缩"= 预测退化为全 idle。）

> **设备与数据根注（重要）**：上表 CPU 行对应 2026-09-03 **CPU** 轮（自动化会话无 GPU），
> GPU 行为 2026-09-04 GPU 轮。
> 2026-09-04 在 GPU（RTX 4060 Laptop / cuDNN 91002 / fp32，数据根 `datasets/cleansight-ActionMixed-auto-lhh`）
> 复跑 roi-grid-144（`runs/formal_roi_20260905/`）：三 seed 零坍缩不变，但逐 seed 漂移显著
> ——seed 42 提升（F1@0.1 33.3→44.9）、seed 7/2026 回落（28.6→19.1、31.8→19.5），中位
> edit/F1@0.1/F1@0.25 = 23.59/19.51/14.63。同日跑完 **GPU 全策略矩阵**
> （`runs/strategy_compare_gpu/`，4 策略 × 3 seed），中位数已并入上表 GPU 行：roi-grid-144
> 仍是唯一跨设备三 seed 零坍缩策略；GPU 上全局 40 零坍缩且中位 F1@0.1 ≈ roi-grid，仅手部
> 40 三 seed 全坍缩（手部 ROI 通道无增益，见 FEATURE_STRATEGY_COMPARE.md 第四轮）。
> **设备差异大于单轮噪声：正式数字必须锚定设备口径，跨设备比较只能定性**。

### 3.4 正式训练方案（2026-09-03 定稿，依据 §3.3 证据）

> **结论先行**：正式训练采用 **ROI 网格 144 维 + 健康配方**；图像通道（§4 E 系列）只在
> 消融证明有段级增益后才并入正式方案——在此之前正式方案不依赖任何图像能力。

| 项 | 定稿 | 依据 |
|---|---|---|
| 特征 | `actionmixed-roi-grid-v1`（144 维） | 三 seed 零坍缩、中位段级指标领先（§3.3）；空间分区特征的稠密性与抗噪性 |
| 数据 | v3（含 task#204 sbc 修正）；action-test 扩量后升版 | 数据同步记录（testsets.yaml 注释 / 6.5 节） |
| 模型 | GRU 滑窗为主（在线推荐基线）；MS-TCN 全序列作对照 | MODELSET_OVERVIEW 流式结论 |
| 配方 | wd=1e-4、dropout=0.2、patience=4、best_metric=val_f1_0.5、epochs≤20、**多 seed** | 坍缩诊断与修复（0059eb9）；单 seed 结论不可靠 |
| 评估 | formal testset、**多 seed 取中位数**、段级指标（edit/F1@0.1~0.5 + 逐类）为准 | acc 在 65%+ idle 数据上具欺骗性 |
| 复跑入口 | `python tools/run_strategy_matrix.py --strategies roi-grid-144 --seeds 42,7,2026` | 一键工具（43f15ef） |

**执行记录**：

- **2026-09-03（CPU 轮，`runs/formal_roi/`）**：健康配方已固化为 `gru-actionmixed-auto-roi.yaml`
  默认值（dropout=0.2 / wd=1e-4 / patience=4 / best_metric=val_f1_0.5，提交后可直接运行无需 -S）；
  正式训练 3 seed 跑通，与多 seed 矩阵轮一致（CPU 环境内确定性复现）：**中位 edit 28.35 /
  F1@0.1 31.8 / F1@0.25 22.7，三 seed 零坍缩**。
- **2026-09-04（GPU 轮，`runs/formal_roi_20260905/`，数据根 `-lhh`，RTX 4060 Laptop）**：
  roi-grid-144 复跑，三 seed 零坍缩，中位 edit 23.59 / F1@0.1 19.51 / F1@0.25 14.63
  （seed 42 升、seed 7/2026 降，见 §3.3 设备注）。同日 GPU 全策略矩阵
  （`runs/strategy_compare_gpu/`，4 策略 × 3 seed）跑完，GPU 口径基线数字定版为 §3.3 表
  GPU 行（roi-grid-144 为唯一跨设备三 seed 零坍缩策略，中位指标见该表）。

## 4. 候选增强实验：bbox 系 + 图像通道（形态 B，E 系列）

### 4.1 融合设计（回应 D1/D2/D3）

```
bbox 特征通道（ROI 网格 144 或基线 40）—— 位置/数量/类别（"哪里有什么"）
        +
图像外观通道：预训练 backbone（冻结）→ 每帧 embedding → 轻量线性投影头（~百~千参数）
        ↓ 拼接（feature_blocks 机制已支持多块声明）→ 时序模型（GRU/MS-TCN…）
```

- **backbone 冻结零训练**（D2/D3）：与 YOLO 角色一致，全链路唯一可训练参数 = 时序模型 + 投影头
- **投影头**：给冻结特征一个领域适配的机会（队友实测：纯冻结 embedding 对下游提升很小，
  "考虑在领域数据上微调"——投影头即最小化的领域适配）；若投影头也学不出增益，才下"图像通道无用"结论
- 因果红线：帧级 CNN 只看当前帧；禁止未来帧/跨帧聚合；推理路径禁随机
- 归一化/预处理（ImageNet mean/std）必须进 feature mapping 契约与版本

### 4.2 实验矩阵（E0 已有，E1~E3 待跑；全部走健康配方 + 多 seed）

| 实验 | 特征组合 | 对照问题 |
|---|---|---|
| **E0（已完成）** | bbox 系四策略（矩阵中位数基准） | 参照系：ROI 网格 F1@0.1 中位 31.8（CPU 轮；GPU 复跑 19.5，见 §3.3 设备注） |
| **E1** | E0 底座 + 整帧 embedding（resnet18 冻结 512 维 → 投影） | 全局外观有无增益 |
| **E2** | E0 底座 + 检测框 ROI 外观聚合（crop_detection 现成） | 干扰过滤后外观有无增益 |
| **E3** | backbone 消融（DINOv2/mobilenet） | 表征质量 vs 成本 |

- 统一验收：段级指标（edit / F1@0.1~0.5 / 逐类），**禁止只报 acc**；与 E0 底座做消融
- 统一配方：wd=1e-4 + dropout=0.2 + patience + 段级 best 指标 + **多 seed 取中位数**

### 4.3 落地步骤（E1 起）

1. **图像源就绪**（§5，阻塞项）
2. 批量提取 embedding（extract_embeddings，GPU 上跑全量）→ 产物目录
3. 登记：embedding 产物作为新数据集契约（testsets.yaml 条目 + feature 声明；
   或按"embedding 目录 + 原 manifest"镜像登记），帧对齐校验进 validate 逻辑
4. `load_split` 增图像契约分支（按视频读 npy，路径可参照 legacy-20d 的 npy 加载方式）
5. 新 recipe/契约常量（维度=bbox_dim + feat_dim，投影在模型层做）
6. 训练配置（E1~E3 各一）+ 一键矩阵扩展策略表
7. 单测（帧缺失/解码失败/维度/确定性）+ validate 门禁
8. 结论写入本文档与 FEATURE_STRATEGY_COMPARE.md

### 4.4 部署影响（重大架构决策，提前知会）

CleanSightBackend 现消费检测框文本派生特征（推理无像素管线）。形态 B 上线需后端新增
"像素 → CNN embedding"管线并与训练期提取口径一致——**不成立则形态 B 只停留在离线评测**。

## 5. 前置条件与阻塞

### 5.1 v3 数据无图（核心阻塞）

v3 auto 仅 `labels/` + `frames/` + `task_ids.yaml`。图像源候选（按优先级）：

| 候选 | 状态 | 备注 |
|---|---|---|
| **LS project-16 下载 18 个原始视频 → stride-4 抽帧** | 凭证在位（.env LS_HOST/TOKEN；上一会话已下载过 task#209 视频），`task_ids.yaml` 提供 task#192~211 完整映射 | 与 v3 manifest 天然对齐；数 GB 下载 + ~1.5~3GB jpg，需网络与存储 |
| **action-test 新视频**（D4） | 待采集 | 走 annotate 时**保留抽帧图**，图像通道从新数据直接可用（v3 同法重建） |
| 手动通道 `cleansight-ActionMixed/images/` | 本地有（606MB） | **与 v3 不同源，禁止混用**；仅作机制测试床（已用于工具冒烟） |
| YOLO 数据集 images | 本地有 | 帧号不对齐，仅检测侧实验 |

### 5.2 数据缺口（D4 背景）

v3 test 只有 idle/long_brush_insert/withdraw 三类——**sb_clean 零覆盖**、water 全库 2 视频。
action-test 的长短毛刷刷洗数据到齐后：`annotate run` → `convert` → 重划 manifest → 重算
revision → validate → 升数据集版本（链路全部现成）。

## 6. 验收与执行顺序

1. **卡点先解**：图像源决策（LS 下载 or action-test 视频路径）
2. 抽帧/对齐工具 + 校验（若走 LS 下载）
3. GPU 全量提取 embedding → 产物登记
4. E1（bbox 底座 + 整帧图像）对照 → 有增益再 E2/E3；无增益按证据收尾
5. 结论 = "本周任务：实验确定图像特征提取方案"的交付（含消融证据、多 seed、段级指标）

## 7. 验证纪律（历史教训，必须遵守）

- **预期管理**：队友实测通用预训练 embedding 对下游提升很小（tracker IDF1 0.4755→0.4806）——
  图像通道收益必须按动作指标验证，禁止预设"有图就更好"
- **坍缩教训**：小数据 + 加权 CE 存在"全 idle 自信输出"低损失吸引子，seed 决定掉进哪个谷——
  所有对照必须多 seed 取中位数，段级指标为主
- **配方健康**：wd + dropout + 早停 + 段级 best 指标（0059eb9 已落地），旧配方结论作废
- **检测契约化**：v3 frames/ 生成时的检测 conf/IoU 阈值未记录——图像/bbox 特征实验前
  应固定并记录（队友推荐 conf 0.25/IoU 0.55 参考）
