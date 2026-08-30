# 时序训练集改造方案（草案）

> **负责人**: 数据集维护（lhh）
> **状态**: 草案 v0.2，§10 决策点已拍板，按定稿执行
> **关联文档**: `docs/DATASET_BUILDING_GUIDE.md`（标注规范）、`docs/AUTO_ANNOTATION.md`（链路细节）、`docs/MODELSET_OVERVIEW.md`（注册规范）
> **范围**: 时序动作训练数据（`cleansight-ActionMixed-auto` 通道）的扩量、重划分、登记与分发全流程

---

## 1. 背景与现状盘点

### 1.1 双通道现状

| 通道 | 数据目录 | testsets 身份 | 框来源 | 状态 |
|---|---|---|---|---|
| 手动通道 | `datasets/cleansight-ActionMixed` | `temporal.actionmixed-v2` | LS 人工画框 | 存量，冻结 |
| 自动通道 | `datasets/cleansight-ActionMixed-auto` | `temporal.actionmixed-auto-v1`（rev `3b1bc00f…`，2026-08-20 起 14 视频） | YOLO 自动检测（`annotate run`） | **本次改造对象** |

特征契约两通道一致：`actionmixed-bbox-8cls-v1`，40 维，6 类动作。

### 1.2 问题清单（按严重度）

| #    | 问题                                                                                                                                                       | 影响                       |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| P0-1 | **test 集单类**：仅 98 帧 `long_brush_withdraw`                                                                                                                | test 指标恒为 0.0，模型真实水平无法评测 |
| P0-2 | **val 缺类**：缺 `air_injection`/`short_brush_cleaning`                                                                                                      | 验证指标失真，早停/选型依据不可靠        |
| P0-3 | **train 类别失衡**：`idle`≈45%，`air_injection`/`short_brush_cleaning` 各≈6%，且训练配置无类别权重                                                                         | 少数类学不出来                  |
| P1-1 | **pin 版本漂移**：`registry/temporal/auto-{gru,mstcn,transformer}-v1/pin.yaml` 钉的是旧 11 视频版 rev `636e6372…`，catalog 已是 14 视频版                                  | 注册信息与现实脱节，复现口径混乱         |
| P1-2 | **auto 通道无 ModelScope 归属**：下载 preset 只有 `yolo`/`actionmixed`/`raw`（`framework/cleansight_eval/core/dataset_download.py::DATASET_PRESETS`），新成员拿不到 auto 数据 | 训练环境搭建断点                 |
| P2-1 | 双生产线并存（`E:\曦源\dataset` 旧构建器为 segment 级 split，且停更于 07-12）                                                                                                 | 有混用风险，需明确边界              |
| P2-2 | dataset 仓库的 `DATASET_STATUS.md` 未登记 auto 通道                                                                                                              | 状态追踪断档                   |

### 1.3 已确认的约束

- 特征映射 `actionmixed-bbox-8cls-v1`（40 维、类别序固定）**不变**——只动数据不动契约；
- 序列长度 ≥ 16 个采样帧（GRU `window: 16`）；
- 同一源视频可跨 split 但帧不得重叠（`split_overlap_policy: frame`）；
- 动作标签必须人工标注，检测框自动生成，二者以文件名匹配合并。

---

## 2. 目标与非目标

**目标**
1. 六类动作在 train 全覆盖且分布改善；val/test 各六类齐全，test 指标恢复可信；
2. 建立"新视频 → 标注 → 自动检测 → 合并 → 重划分 → 登记"的可重复维护流程并文档化;
3. auto 通道可一键下载（ModelScope + CLI preset），注册信息与数据一致。

**非目标**
- 不改特征映射/类别表（若未来增删动作类，另立全链路工程方案）;
- 不回溯重建手动通道 v2;
- 不动 YOLO 检测器本身的训练数据（`cleansight-yolo` 线）。

---

## 3. 总体路线

```
Phase 0 基线准备 ──► Phase 1 数据扩量 ──► Phase 2 split 重划与登记 ──► Phase 4 分发托管 ──► Phase 3 模型联动
   （本周）            （随新视频）          （数据到位后立即）        （数据稳定后）        （后续轮次，决策 D）
```

依赖关系：Phase 2 必须等 Phase 1 的首批新视频进库；Phase 4 依赖 Phase 2 定稿（避免传一版很快过期的数据）；Phase 3 本轮暂缓（决策 D），数据侧完成后另起一轮补上。

---

## 4. Phase 0 — 基线与准备（先行，不依赖新视频）

1. **数据本体到位**：本机目前没有 auto 通道数据。从持有数据的机器同步 `datasets/cleansight-ActionMixed-auto/` 与 `outputs/annotations/`（或等 Phase 4 的 preset 就绪后走标准下载）。
2. **出基线分布表**：统计当前 train/val/test 六类帧数分布（现有数字来自指南口述，改造前后必须有精确表格对照）。产出物存 `benchmark/reports/` 或随 PR 附表。
3. **冻结旧生产线**（决策 B）：在 dataset 仓库 README 标注"actionmixed 手动框构建器不再接收新视频，新数据一律走 auto 链路"；构建代码**保留不删**（保证历史版本可复现），仅停止使用。

**验收**：基线分布表完成；本机能跑通 `python tools/validate_testsets.py --catalog framework/testsets.yaml --json` 且全部 OK。

---

## 5. Phase 1 — 数据扩量

### 5.1 采集要求（给采集同学）

- 优先覆盖少数类片段：**`air_injection`、`short_brush_cleaning`、`long_brush_withdraw`**；
- 一个视频 = 一个连续时间序列；命名沿用 `<8位hex>-clip_<起始ms>_<结束ms>.mp4`，全局唯一；
- 单视频采样后帧数 ≥ 16（不足的不收）。

### 5.2 标注（给标注同学）

按 `docs/DATASET_BUILDING_GUIDE.md` §2–§5 执行：只标 timelinelabels 动作区间，不画框；标签名严格六选一；导出 JSON 文件名带日期。

### 5.3 自动检测与合并（维护人执行）

```bash
# ① 新视频批量检测（断点续跑）
python -m framework.cleansight_eval.cli.annotate run \
    --videos <新视频目录> --config framework/experiments/auto-annotate.yaml \
    --frame-stride 4 --resume

# ② 合并人工动作标签（train 先行；val/test 在 Phase 2 划定后跑）
python -m framework.cleansight_eval.cli.annotate convert \
    --annotations outputs/annotations \
    --labels-export <最新人工导出.json> \
    --out datasets/cleansight-ActionMixed-auto --split train
```

### 5.4 resume 陷阱检查单（历史事故：05ba4406/4807dbbe 全零）

- [ ] 本次 run 之前没用过 `--max-frames` / 改过 `--out`；用过则先删除 `outputs/annotations/` 中对应旧 JSON 再续跑
- [ ] convert 输出结尾**无任何"跳过"告警**（有跳过 = 该视频没进数据集）
- [ ] 抽样画框预览至少 3 个新视频：`python tools/visualize_annotations.py ...`
- [ ] 新视频序列长度 ≥ 16 采样帧

---

## 6. Phase 2 — split 重划与登记（核心环节）

### 6.1 重划策略

- **原则**：以"每类在三个 split 都有足够样本"为第一目标；同一源视频的帧不跨 split（沿现行 `frame` 策略）；
- **test 修复**（P0-1）：必须动，当前恒零无锚点价值。变更原因写入 commit message 与本节（对齐 2026-08-21 只动 train 的留痕惯例的反向操作：这次动 test，理由充分）；
- **val 处理**（P0-2）：**整体重划**（决策 A，已定）——与 test 一并按类别比例重新划分三个 split。理由：当前数据量小、重训成本低，且 Phase 3 暂缓后本轮没有历史指标包袱；重划后旧 val/test 指标不再可比，此事实写入 commit message 与变更记录。
- 划分结果落 `benchmark/manifests/actionmixed-auto/{train,val,test}.txt`（auto 通道有独立的 manifest 目录，勿与手动通道的 `actionmixed/` 混用）。

### 6.2 登记三件套（每次数据变更必做，顺序固定）

1. 更新 `actionmixed-auto` 三份 manifest；
2. 重算并更新 `framework/testsets.yaml` 中 `temporal.actionmixed-auto-v1.revision`（三个 manifest 拼接内容的 sha256）；**同一次改动中同步更新 `usage/YAML_CONFIG.md`**（仓库规定：tracked YAML 变更必须同改该索引）；
3. `python tools/validate_testsets.py --catalog framework/testsets.yaml --json` 全部 OK。

**验收**：validate 全 OK；分布表显示 train/val/test 六类齐全（个别类偏少需显式备注）；revision 与 manifest 一致。

### 6.3 执行结果（2026-08-27，已完成）

- 数据集 `temporal.actionmixed-auto-v2` 重建完成：26 视频，train 17 / val 5 / test 4，revision `c65f26b8…`
- 六类覆盖（标签帧）：test 每类 115~770；val 每类 119~386；train 保留 air 大户(af4ea419 458)与 flush 大户(63a848d5 770)
- `validate_testsets.py`：auto-v2 三 split 全部 OK（fingerprint 已生成）；仅存的 5 个 FAIL 为本机未挂载的历史数据集（手动 v2 / endo），非本次回归
- catalog/实验配置/YAML_CONFIG 已同步升 v2；3 个未质检候选已按方案 a 合入（37c53d37→val、f4b10ad8/3b2dcda0→train），质检不通过时走 manifest 剔除流程

---

## 7. Phase 3 — 模型侧联动（本轮暂缓，决策 D）

> 本阶段本轮不执行，优先完成数据侧（Phase 0–2、4）。**一项轻量工作建议立即做**：在 v1 三个模型的 CARD.md 补注"训练数据为 11 视频旧版 rev `636e6372…`"，避免 Phase 3 补做前有人拿 v1 指标与新数据混比。

1. 用重划后的数据重训 auto 系三模型（gru / mstcn / transformer），配置 `framework/experiments/*-actionmixed-auto.yaml`；
2. 注册新版本：`registry/temporal/auto-*-v2/`（CARD.md + pin.yaml + 评测报告），pin 的 dataset revision 指向 Phase 2 的新 revision，**同时修正 v1 的漂移说明或在 v1 CARD 注明其训练数据为 11 视频旧版**；
3. 特征映射不变 ⇒ 不需要动 CleanSightBackend 的 feature 层；上线前按惯例做 backend 集成冒烟。

**验收**：新模型在修复后的 test 上出全六类指标；v2 pin 校验通过；v1 漂移有书面交代。

---

## 8. Phase 4 — 分发托管

### 8.1 ModelScope 新建 auto 仓库（dataset 仓库侧操作）

1. ModelScope 建仓 `lhh010/cleansight-ActionMixed-auto`；
2. `dataset/config.py` 增加 `MS_ACTIONMIXED_AUTO_REPO_ID`（该文件不入 git，模板 `config.example.py` 同步加注释行）;
3. 仿照 `cleansight-pipeline/actionmixed/upload.py` 写 `upload_auto.py`，上传内容与 auto 通道目录结构对齐（`labels/` + `frames/` 等，以 convert 实际产物为准）；
4. 仓库描述注明对应的 revision hash；
5. **首版上传执行前须经数据集负责人确认数据版本与 revision（决策 C 的确认环节）**。

### 8.2 本仓库下载接入

- `framework/cleansight_eval/core/dataset_download.py`：`DATASET_PRESETS` 增加 `actionmixed-auto`，`REQUIRED_FILES` 增加对应校验文件；
- `framework/cleansight_eval/cli/dataset.py`：`--preset all` 的循环 keys 加入新键；
- `datasets/README.md` 与 `TEAM_GUIDE.md` 的数据清单补充 auto 通道条目。

### 8.3 状态登记补全（dataset 仓库）

`DATASET_STATUS.md` 增加"时序 auto 通道"小节，登记每次 revision 变更日期、视频数、六类分布摘要。

**验收**：新机器 `python -m framework.cleansight_eval.cli.dataset --preset all` 后能直接跑通任一 auto 实验的训练冒烟。

---

## 9. 红线清单（任何阶段都不可违反）

- [ ] 已标注视频文件永不替换/重编码（帧号错位 = 标签作废）；
- [ ] 检测结果只当特征，动作判定永远以人工标签为准（检测对 `short_brush`/`brush_tip_out` 恒零是已知限制）;
- [ ] 数据内容变化必须走登记三件套，禁止只改数据目录;
- [ ] testset 变更必须在 commit message / 文档留痕变更原因;
- [ ] tracked YAML 变更同改 `usage/YAML_CONFIG.md`;
- [ ] 视频、原始 LS 导出不入 Git。

## 10. 决策记录（已定）

| #   | 决策                   | 结论                                          |
| --- | -------------------- | ------------------------------------------- |
| A   | val 修复方式             | **整体重划**——数据量小、重训快；接受旧 val/test 指标不可比并留痕    |
| B   | 旧手动通道                | **不再接收新视频，代码保留**（冻结不删除，保证历史可复现）             |
| C   | auto 上 ModelScope 时机 | **Phase 2 定稿后一次性上传首版；上传执行前需负责人确认**          |
| D   | Phase 3 本轮是否执行       | **暂不执行**，优先数据侧，后续轮次补上；v1 pin 漂移先以 CARD 注记过渡 |

## 11. 线路 B 执行记录（2026-08-25）

> 决策：本机无 legacy v3 权重，改用本机 yolo11 实验权重独立重建检测标注（annotation source 变更，升数据集版本登记）。

- 权重：`EXPERIMENTS/best_weights/yolo11s-g{1,2}-default-best.pt`（default 增强变体）→ 已复制入 `legacy/yolo-detection/pipeline/versioned_weights/yolo11s-g{1,2}-v1/best.pt`；g2 覆盖 5 类小目标，**short_brush/brush_tip_out 特征不再恒零**（相对 v1 数据是特征语义增强，重训时需关注）
- 配置：新建 `framework/experiments/auto-annotate-yolo11.yaml`（out_dir `outputs/annotations-yolo11`，与 legacy 产物 `outputs/annotations` 隔离）；`usage/YAML_CONFIG.md` §8 已同步登记
- 视频：26 个已有人工动作标注视频（manifest 14 + 扩量候选 13 含未质检，检测与质检解耦，未质检者不进 Phase 2），硬链接暂存 `outputs/videos-auto26`
- 环境：系统 python + ultralytics 8.4.130（CPU 推理）；冒烟已通过（137 帧 4 轨迹）
- **检测完成（2026-08-27）**：26/26 JSON 产出至 `outputs/annotations-yolo11`，终检通过——无空轨迹视频，hand×2 全局一致，8 类全覆盖（hand 52 / control_body 26 / mid_section 26 / short_brush 18 / syringe 18 / distal_end 18 / air_gun 15 / brush_tip_out 5），帧覆盖与视频长度一致
- 质检状态基线：见 `E:\曦源\dataset\raw-from Label Studio\EXPORT_NOTES.md`
- 暂存记录：`687e3c78`（在库但无动作标注）已暂存至 `outputs/stash-pending-videos/`（仅本地，不外发；处置与流程见 EXPORT_NOTES.md 第四节）

## 11A. v3 重建（2026-08-28，project-16 数据源切换）

应团队变更要求完成 v3 重建，关键变更：

- **数据源切换**：LS project-16（18 个 8 月新录视频，task#192-211），旧 project-10 的 26 个视频不在新导出内；v2 产物备份于 `datasets/cleansight-ActionMixed-auto-v2-backup`
- **标签与 LS 同步**：`air_injection` 更名 `water_injection`（action id 位置 1 不变）；`_constants.py::ACTION_CLASSES` 已同步
- **task id 溯源**：数据集根新增 `task_ids.yaml`（split → 视频 → LS task id 映射）
- **test 锚定**：按团队指定 = LS task#195（5b181b9b）/ #199（1b2c95ff）；注意 test 仅含 insert/withdraw 两类，water/flush/sbc 不在 test 中
- **split**：train 13（9142 帧，六类齐）/ val 3（1926 帧，五类齐；water 仅 17 帧——全库仅 2 个 water 视频所致，152453e5#207 在 val、39da2635#201 在 train）
- **revision**：`b7edb874…`；catalog/experiments/YAML_CONFIG 已升 v3；validate 全绿
- **检测**：18/18 yolo11 双模型完成（outputs/annotations-yolo11 追加 18 份 JSON）
- **ModelScope**：`lhh010/cleansight-ActionMixed-auto` 仓库已建，上传脚本已备（`dataset/cleansight-pipeline/actionmixed/upload_auto.py`），**执行前需负责人确认**

## 11B. v3 质审更新（2026-08-30）

- task#195/199/211 审核通过，标注无变化；**task#204 审核修正 4 段 sbc 区间**，已过滤式重建（sbc 39→48 帧），标签文件已同步 ModelScope，manifest/revision 不变（split 未动）
- 新导出发现 **task#203**（f809e944，water_injection 177 帧）未审，暂不入库，待 QC
- 质审台账：dataset 仓库 raw-from Label Studio/EXPORT_NOTES.md 第六节

## 11C. 数据开发优先级参考（2026-08-30 引入）

依据 `dataset/docs/BENCHMARK_SEGMENTATION.md`（benchmark 等价类划分）的优先级框架，指导后续数据采集与扩量方向：

**采集优先级（对数据侧的含义）**
- **P0（缺口类/对抗序列，需真实采集，增强造不出）**：对抗-可见性序列（器具在场未操作 / 多器具同框仅一件在用）、insert vs withdraw 对比段、模糊渐变边界、罕见转移对、段内遮挡/手离场
- **P1**：超短/超长段、动作重复/缺失 phase、无停顿衔接、长 idle、异机位
- **P2/已饱和（作对照基线，无需扩量）**：**flush、long_brush_insert**
- 类别缺口口径与 v3 现状对照：文档标注 air(现 water)_injection/sbc 为 P0 缺口是 benchmark test 策展视角；**维护人 2026-08-30 决定：water_injection 扩量优先级调低**，task#203(f809e944) 暂缓入库，sbc 仍按缺口对待

**对 split/评测纪律的启示（v3 已符合）**
- test 源级整条隔离（v3:视频级 split，无跨 split）✓
- test 整段保留不截散帧（v3:整视频入库）✓
- 后续 Phase 3 重训时，评测报告建议按文档 §7 做 EC 切片指标（对抗-可见性子集 vs 普通子集 F1 衰减等）

## 12. 附录：命令速查

见 `docs/DATASET_BUILDING_GUIDE.md` §5–§6 与本方案 §5.3、§6.2。常用校验：

```bash
python tools/validate_testsets.py --catalog framework/testsets.yaml --json   # 目录/清单校验
python -m framework.cleansight_eval.cli.dataset --check                      # 数据就绪校验
python -m framework.cleansight_eval.cli.dataset --list-presets               # 数据源清单
```
