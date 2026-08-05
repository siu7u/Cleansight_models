# CleanSight YOLO · 数据与模型侧工作汇报

> 汇报日期：2026-08-04 ｜ 范围：近几天 YOLO 检测数据建设与模型训练/评测进展
> 数据来源：仓库本地磁盘实测（`datasets/cleansight-yolo/`），非 README 快照

## 一句话总结

新标准 YOLO 数据集已就位（训练轨 + benchmark 评测轨双轨，含 test split），训练/评测链路已跑通并工具化；下一步是在 GPU 上重建新数据基线，并针对小目标与稀有类做专项优化。

---

## 一、数据侧进展

### 1.1 新数据集落盘（`datasets/cleansight-yolo/`，ModelScope `lhh010/cleansight-yolo`，约 3.8GB）

- **双轨分离**：训练轨（LS `yolo-train` 项目，24 个任务）产出 `train`/`val`；评测轨（LS `yolo-test` 项目，42 个任务，含场景 tag/等价类策展）产出独立 `test`。
- **切分契约**：按 LS 任务整段切分，同一任务的所有帧只进一个 split，杜绝时间相邻帧泄漏；`val` 手动按类覆盖分配（补齐稀有类）。
- **构建管线**：关键帧对齐 + 线性插值、`stride=4` 抽帧（≈7.5 张/秒）、稀有类相邻帧密采、空帧丢弃。

### 1.2 样本规模（磁盘实测）

| 组 | 类别数 | train | val | test | 合计 |
|---|---:|---:|---:|---:|---:|
| group1_large（大目标） | 3 | 21,526 | 5,169 | 6,786 | 33,481 |
| group2_small（小目标） | 5 | 14,916 | 4,746 | 3,592 | 23,254 |
| 合计 | 8 | 36,442 | 9,915 | 10,378 | 56,735 |

### 1.3 类别与框分布（train 框数）

| 组 | 类别 | 框数 |
|---|---|---|
| group1_large | hand | 38,609 |
| | scope_control_body | 17,039 |
| | scope_mid_section | 17,100 |
| group2_small | syringe | 7,116 |
| | air_gun | 1,830 |
| | scope_distal_end | 9,642 |
| | short_brush | 1,804 |
| | brush_tip_out | 572 |

> ⚠️ test 中稀有类仍稀疏：`brush_tip_out` 仅 21 框、`air_gun` 293 框 —— 评测置信度受限。

### 1.4 数据特性结论（指导建模）

- **小目标突出（group2）**：58% 的框面积 < 图面积 1%（中位 0.43%）→ 需要高分辨率输入/小目标专项。
- **类别不均衡（group2）**：`air_gun`/`short_brush`/`brush_tip_out` 训练框数仅数百~1.8k，相对 `scope_distal_end`(9.6k) 差距大 → 需要类别加权/稀有类增强。
- **视频帧高度相关**：抽帧+密采导致相邻帧近似重复 → 有效信息量小于文件数，注意防过拟合。
- group1 目标较大（框面积中位 3.1%），3 类相对均衡。

### 1.5 数据工程产出

- 修正各分组 `data.yaml`（路径解析修复后，仓库根可直接训练/评测）。
- 产出 `tracking_train.md` / `tracking_test.md`（任务↔split 追踪）、`tag_metadata.json`（评测轨场景 tag）。
- 数据集下载脚本 `download_modelscope_dataset.py --preset yolo` 可用。
- ✅ **已解决**：`framework/testsets.yaml`（catalog）已登记新数据集 `datasets/cleansight-yolo/`
  （group1 3 类 / group2 5 类），`tools/validate_testsets.py` 校验通过；旧 `datasets/yolo/` 已删除。

---

## 二、模型侧进展

### 2.1 环境

- torch 2.8.0 + ultralytics 8.3.253（backend venv）。
- 目标机具备 CUDA；当前 agent 沙箱会话无 GPU 直通（`/dev/dxg` 不可见），正式训练在目标机执行。

### 2.2 已完成

- **冒烟链路跑通**：yolo11n 在 group1 上完成 train→val→test 全链路（2 epoch / imgsz 416 / 5% 采样），产物 `runs/cleansight-yolo/smoke-g1-cpu/`（results.csv、PR/mAP 曲线、best.pt）。
- **优化工具 framework 化**：`tools/yolo_smoke.py` 等独立优化脚本已移植进 framework/benchmark
  —— `cli.sweep`（多预设/grid 实验）、`cli.analyze`（逐类淘汰决策）、`roi_classification` 流水线
  （特征融合）；`tools/` 不再维护优化脚本（详见 `docs/YOLO_OPTIMIZATION.md`）。
- **benchmark 评测链路打通**（exploratory 模式）：`python -m benchmark.cli.eval --config ... --ckpt ...` 可产出含逐类 P/R 的评估报告。
- **职责分离**：catalog（`framework/testsets.yaml` → `core/catalog.py`）与指标原语（`core/metrics.py`）
  下沉到 framework，依赖方向单向 `benchmark → framework`，framework 生产代码不再 import benchmark。
- **旧模型交叉验证**：旧数据训练的 `clean-large-v0.2` 在新 val 上 mAP@0.5 = 0.512（旧口径下为 0.935）→ 证实新旧数据分布漂移，旧权重不可作为新基线。

### 2.3 现状指标（仅链路验证，非正式结果）

| 项 | 值 | 说明 |
|---|---|---|
| 冒烟训练（2 epoch，5% 采样） | val P 0.44 / R 0.43 / mAP50 0.36 | 仅验证流程，不可引用 |
| 旧模型 × 新数据 | mAP50 0.512 / mAP50-95 0.183 | 分布漂移对照 |

### 2.4 已知问题

- ✅ **已解决**：integrity 校验曾因 `testsets.yaml` 未同步登记而红；catalog 迁移后
  `tools/validate_testsets.py` 全绿（YOLO 条目）。
- 新数据上**尚无正式 baseline**。

---

## 三、下一步计划（按优先级）

1. ✅ **修评测口径**：`framework/testsets.yaml` 已登记新数据集条目（group1 3 类 / group2 5 类）→ `tools/validate_testsets.py` 校验通过。
2. **重建基线**（GPU）：`python -m framework.cleansight_eval.cli.sweep --group group1_large --preset large_baseline`
   （yolo11n + 全量数据 + imgsz 640）与 `small_baseline`（group2 / 960），以 val 为准记录
   mAP50 / mAP50-95 / 逐类 P·R。
3. **小目标专项**：group2 提升分辨率（`small_s_1280_p2` 等预设）；必要时 P2 高分辨率特征头。
4. **稀有类专项**：`copy_paste`/`mixup` 增强（`small_s_copy_paste`），对照逐类 recall。
5. **模型规模对比**：sweep grid（models × resolutions）消融，记录参数量/速度/精度权衡。
6. **淘汰决策**：`python -m benchmark.cli.analyze --config framework/experiments/yolo-clean-small.yaml --ckpt <best.pt>`，
   P/R < 0.3 的类从 YOLO 淘汰 → `roi_classification` 特征融合（`framework/experiments/roi-fusion.yaml`）。
7. test split 仅在最终验收使用。

---

## 汇报口径速记

- **30 秒版**：新数据集双轨就绪（5.6 万图 / 8 类，含独立 test），训练评测链路跑通；优化工具已
  framework 化（sweep/analyze/roi_classification）；本周目标 —— 新基线重建 + 小目标/稀有类优化
  + 淘汰决策。
- **一页版**：见上文各节，核心数字取 §1.2 样本表、§2.3 指标表。
- 所有指标标注"口径/条件"（采样比例、split、设备），避免新旧数据指标混比。
