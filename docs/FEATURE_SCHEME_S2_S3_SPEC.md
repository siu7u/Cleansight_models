# S2/S3 特征方案实施规格（feature-scheme S2/S3 spec）

> 上游：[`FEATURE_SCHEME_EVAL_PLAN.md`](FEATURE_SCHEME_EVAL_PLAN.md)（方案总计划）·
> 本文是 **S2（bbox ⊕ 全帧 CNN embedding）与 S3（手部 ⊕ 全局 bbox ⊕ 全帧 CNN）的具体工程实施规格**，
> 含数据现实约束、维度设计、登记清单、执行命令与验收判据。实现落点与结果回写见
> [`FEATURE_LAB.md`](FEATURE_LAB.md)。

## 0. 一句话

在已完成的四特征集矩阵（ROI 144 最优、纯手部最差，见 FEATURE_LAB §6）基础上，引入
**全帧 CNN embedding** 作为全局外观特征，验证"视觉外观信息能否超越纯几何 bbox"，并
重新检验手部通道在外观特征加持下是否有增益。

## 1. 前置事实与约束（实施前必读）

1. **数据集不含图片**（`datasets/cleansight-ActionMixed-auto/README.md §一`）：只有
   `labels/`（动作标签）与 `frames/`（YOLO 检测 txt）。S2/S3 必须先解决**帧图来源**。
2. v3 源视频为 LS project-16 的 18 个 2026-08 新录视频，映射见 `task_ids.yaml`
   （train 14 / val 4）。本机检索 `outputs/videos-auto26/` 等目录均为 2026-02 旧素材，
   **不匹配**；源视频需从 LS 导出或录制端原始目录获取（待办，见 §6 步骤 0）。
3. 帧对齐口径：标签行 frame_id 为 **1-based 真实视频帧号**（按 ~7.5fps 均匀采样），
   帧图文件名必须与 `frames/<split>/<视频>.mp4-<帧号:06d>` 的帧号一一对应。
4. 评测纪律不变：新特征契约 = 新 feature_mapping 版本；exploratory（CPU/WSL）口径标注；
   3 seed 中位数排序；判据仍是「同时优于 B0a(bbox-40) 与 B0b(roi-144)」。

## 2. 特征契约与维度设计

resnet18（ImageNet 预训练、去分类头）逐帧 embedding 为 512 维。为复用现有
catalog 的 `feature_blocks` 校验（input_dim = 检测类数 8 × 5 × blocks），投影维度定为 **80**：

| 契约 | feature_mapping | 组成 | 维度 | blocks |
|---|---|---|---:|---:|
| S2 | `actionmixed-bbox-cnn-resnet18-v1` | bbox-40 ⊕ PCA-80(embedding) | **120** | 3 |
| S3 | `actionmixed-bbox-hand-cnn-v1` | bbox-40 ⊕ hand-40 ⊕ PCA-80(embedding) | **160** | 4 |

- PCA 在 **train split 的全部帧 embedding** 上拟合（numpy SVD，零均值），均值向量与
  成分矩阵存 `pca80.npz`（含 meta：backbone/输入尺寸/拟合帧数/sha256），val 推理只用
  该固定变换——保证因果性（val 不参与拟合）与可复现。
- 投影后向量做与 bbox 同式的有界化（clip 到 [-5, 5] 后线性缩放到 [0,1]），避免量纲失衡；
  具体口径写入 npz meta 并固化在单测中。
- 缺图帧（提取失败/跳帧）embedding 补零并记 `imputed` 语义（与 bbox 契约空帧全零一致），
  统计写入 meta，超过 5% 的视频在报告中单独标注。

## 3. 帧图与 embedding 预计算管线

### 3.1 帧图提取（新工具，落点 `tools/extract_sampled_frames.py`）

- 输入：源视频目录 + 数据集 root + split 列表；
- 对每个视频读取 `labels/<split>/<视频>.mp4.txt` 的 frame_id 集合，用 ffmpeg/OpenCV
  按 1-based 帧号精确抽取，存 `datasets/cleansight-ActionMixed-auto/images/<split>/<视频>.mp4/<帧号:06d>.png`；
- 帧图目录 gitignore（不入库，体积大）；产出 `images_manifest.json`（逐视频帧数/缺失清单）。

### 3.2 embedding 预计算（复用已迁移的 `temporal/features/extract_embeddings.py`）

```bash
python -m framework.cleansight_eval.temporal.features.extract_embeddings \
    --root datasets/cleansight-ActionMixed-auto \
    --splits train val --backbone resnet18 \
    --out-dir runs/image_embeddings/actionmixed-resnet18-v1
```

- 产物：`<out>/<split>/<视频>.mp4.npy`（`[T, 512]` float32，与标签行一一对齐）+ `meta.json`；
- 因果、确定性（eval+no_grad+无随机），CPU 可跑（resnet18 ~10 fps，全量约 4~5k 帧预计 <15 分钟）。

### 3.3 PCA 投影拟合（新脚本，落点 `tools/fit_embedding_pca.py`）

读 train 全部 npy → 拟合 PCA-80 → 写 `runs/image_embeddings/actionmixed-resnet18-v1/pca80.npz`。

## 4. 代码与登记清单

| 项 | 位置 | 要点 |
|---|---|---|
| S2/S3 recipe | `temporal/features/cnn_concat.py`（新） | 读 npy + pca80.npz，与 bbox/hand 40 维拼接；无状态、逐帧对齐 |
| data.py 装配 | `temporal/data.py` | 新增两个 feature_version 分支：npy 缺失时显式报错（不静默回退 bbox） |
| testsets 登记 | `framework/testsets.yaml` | `temporal.actionmixed-auto-cnn-v1`（input_dim 120, blocks 3）、`temporal.actionmixed-auto-hand-cnn-v1`（160, blocks 4）；data_root/revision 与 v3 相同，无 test split |
| 实验 YAML | `framework/experiments/gru-actionmixed-auto-{cnn,hand-cnn}.yaml` | 复制 roi yaml 骨架，改 input_dim/feature_schema/dataset_ref |
| 单测 | `framework/tests/test_cnn_concat.py` | 对齐、缺帧补零、PCA 复现性、mask_targets 语义 |
| feature_names | `temporal/features/__init__.py` | 登记两个新版本名 |

## 5. 实验矩阵（增量）

- 跑法与第一轮相同：`gru-{cnn,hand-cnn}.yaml` × seed 42/7/2026，20 epoch 健康配方，
  训练完接评估（复用 `tmp/run_feature_matrix.py` 的编排逻辑，扩充 CFGS 即可）；
- 汇总表在 FEATURE_LAB §6 基础上追加两行，六特征集同表排序。

## 6. 执行步骤

0. **获取源视频**（当前阻塞项）：从 LS project-16 导出 18 个视频（或录制端目录同步），
   放 `outputs/source_videos_project16/`；逐个核对文件名与 task_ids.yaml 一致。
1. 帧图提取（§3.1）→ 核对 images_manifest.json 缺失率。
2. embedding 预计算（§3.2）→ PCA 拟合（§3.3）。
3. 代码与登记（§4）→ 单测全绿 → testsets 校验通过。
4. 冒烟：`gru-actionmixed-auto-cnn.yaml` 2 epoch 跑通。
5. 增量矩阵 6 run + 评估 → 汇总回写 FEATURE_LAB §6。

## 7. 验收判据与风险

- **S2 成立**：median edit 与 F1@0.25 同时 > roi-144（24.83 / 18.18）→ 引入外观特征的
  正式版本（GPU 复跑确认后升 feature_mapping 正式候选）。
- **S3 成立**：S3 > S2 且 S3 > roi-144 → 手部通道在外观特征加持下有增量；若 S3 ≈ S2，
  判定 box 锚定手部通道无增益，S1 线正式关闭。
- 风险：①源视频不可得 → S2/S3 阻塞（唯一硬阻塞）；②ImageNet 特征与内镜域差距大 →
  若 S2 无增益，二期做 linear-probe 微调再评（不在本轮）；③PCA-80 信息损失 → meta 中
  记录解释方差比，<85% 时考虑 128 维（blocks 语义改为独立校验分支）。
