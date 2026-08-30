# YOLO 仓库工作交接更新文档

更新时间：2026-08-30
当前分支：`feat/yolo-optimize-gl`
基线分支：最新 `origin/feat/yolo-optimize`
当前状态：本轮 YOLO 训练、后处理、tracker、ROI backbone 相关代码和文档随本分支交付；`runs/`、`datasets/`、权重文件由 `.gitignore` 忽略，交接时需要单独保留本地实验产物。

## 交付版文档入口

原 handoff 已按所属关系拆分为可逐项复现的实验文档，所有路径均以仓库根目录为基准：

- `README.md`：整个模型仓库结构、框架入口和本文档索引。
- `experiments/README.md`：个人新增/修改内容的边界说明。
- `experiments/yolo/README.md`：按日期排列的 YOLO 完整时间线。
- `experiments/yolo/training/`：三次模型训练的设计、命令、指标和结论。
- `experiments/yolo/postprocess/`：单帧后处理、轨迹 GT、BoT-SORT 调参和 val/test 复验。
- `experiments/yolo/performance/`：GPU 延迟和 ROI backbone 对比。

`experiments/` 只承载个人实验文档，不移动或重排远端已有的 framework、benchmark、registry、legacy 内容。
各叶子 README 都包含脚本路径、命令格式、参数解释、输入输出、结果、分析和下一步计划。

## 1. 本次整理

已删除明确的 smoke 文件和临时 smoke 产物：

| 类型 | 路径 | 处理 |
| --- | --- | --- |
| tracked 测试 | `framework/tests/test_detection_smoke.py` | 删除 |
| tracked 测试 | `framework/tests/test_mstcn_smoke.py` | 删除 |
| tracked 测试 | `framework/tests/test_pipeline_smoke.py` | 删除 |
| ignored 临时目录 | `runs/gpu_latency_bench_smoke/` | 删除 |
| ignored 临时目录 | `runs/roi_backbone_tracker_compare_smoke/` | 删除 |
| ignored 临时目录 | `runs/smoke_configs/` | 删除 |
| 本地缓存 | `.pytest_cache/`、仓库根目录 `__pycache__/` | 删除 |

没有删除 `benchmark` 结果 schema 里的 `limits.is_smoke` 字段，也没有清理历史文档中对 smoke 概念的说明。原因是这些字段属于统一 benchmark 结果结构的一部分，直接删除会影响旧结果解析和 schema 兼容。

## 2. 未提交代码变更概览

| 文件 | 变更内容 |
| --- | --- |
| `framework/cleansight_eval/cli/train.py` | 新增 `--run-id` 参数，支持显式指定训练 run 目录名。 |
| `framework/cleansight_eval/core/pipeline.py` | `Pipeline.train()` 抽象接口增加 `run_id` 参数。 |
| `framework/cleansight_eval/classification/pipeline.py` | 透传 `run_id` 到 `RunContext`。 |
| `framework/cleansight_eval/detection/pipeline.py` | 透传 `run_id` 到 `RunContext`。 |
| `framework/cleansight_eval/temporal/full_sequence_pipeline.py` | 透传 `run_id` 到 `RunContext`。 |
| `framework/cleansight_eval/temporal/sliding_window_pipeline.py` | 透传 `run_id` 到 `RunContext`。 |
| `framework/cleansight_eval/core/run.py` | 对显式 `run_id` 做目录名安全校验，只允许字母、数字、点、下划线、短横线。 |
| `framework/cleansight_eval/core/config.py` | 将 Ultralytics YOLO 的训练超参数加入已知配置键，避免配置校验误报。 |
| `framework/cleansight_eval/detection/yolo.py` | 新增 YOLO 可透传训练参数白名单，并把 optimizer、lr、warmup、loss、augmentation 等参数传给 `YOLO.train()`。 |
| `framework/tests/test_run_status.py` | 新增 `RunContext` 拒绝路径型 `run_id` 的测试。 |
| `tools/visualize_detections.py` | 增加仓库根目录导入修正；视频可视化新增 `max_frames` 限制，便于只渲染片段预览。 |

新增工具脚本：

| 文件 | 用途 |
| --- | --- |
| `benchmark/cli/postprocess_detection.py` | 对已保存 YOLO prediction artifact 做离线后处理评测，支持阈值、NMS、Soft-NMS、WBF、top-k、面积过滤、逐类指标和 split CSV。 |
| `tools/evaluate_labelstudio_trackers.py` | 使用 Label Studio 原生 `videorectangle.sequence` 恢复 GT track id，并评估 YOLO tracker 的 IDF1、ID Switches、Fragments 等轨迹指标。 |
| `tools/render_tracking_comparison.py` | 渲染 track 前/track 后或三栏对比视频，支持轨迹尾迹显示。 |
| `tools/explore_yolo_postprocess_next.py` | 用 val 选 YOLO 后处理参数，再在 test 上验证；比较 `imgsz=640/768` 和多种后处理策略。 |
| `tools/benchmark_gpu_roi_track_latency.py` | 分开测试 ROI 特征提取耗时与 YOLO 检测后 tracker update 耗时，输出 GPU、显存、帧数、batch 等报告。 |
| `tools/compare_roi_backbones_for_tracking.py` | 对比多种 ROI backbone 作为 BoT-SORT ReID 特征时的速度、显存、IDF1、ID Switches、F1 等指标。 |

新增交付文档：

| 路径 | 用途 |
| --- | --- |
| `experiments/README.md` | 说明个人内容边界，避免误动远端基础内容。 |
| `experiments/yolo/README.md` | 按时间汇总全部 YOLO 实验和当前推荐方案。 |
| `experiments/yolo/training/` | 三次训练实验的逐项复现文档。 |
| `experiments/yolo/postprocess/` | 后处理与 tracking 的逐项复现文档。 |
| `experiments/yolo/performance/` | GPU/ROI 性能实验的逐项复现文档。 |

## 3. 按时间顺序的工作记录

### 2026-08-06：YOLO11s strong-cos 完整训练与正式评测

目标：以 `yolo11s.pt` 为基础，在 `group1_large` 上进行完整训练，尝试较强数据增强和 cosine 学习率。

配置来源：

- `runs/optimized_configs/yolo11s-group1-large-strong-cos.yaml`
- resolved 配置：`runs/yolo-20260806-135408/config.resolved.json`

核心训练参数：

| 参数 | 值 |
| --- | --- |
| weights | `yolo11s.pt` |
| imgsz | `640` |
| epochs | `30` |
| batch | `16` |
| patience | `10` |
| cos_lr | `true` |
| close_mosaic | `10` |
| mosaic / mixup | `1.0 / 0.15` |
| scale / translate / shear | `0.7 / 0.2 / 2.0` |

产物：

- 最佳权重：`runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt`
- 最后一轮权重：`runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/last.pt`
- 训练曲线：`runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/results.png`
- 训练日志 CSV：`runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/results.csv`
- 正式评测：`runs/yolo-20260806-135408/evals/detection-yolo-20260806-153902.evaluation.json`
- 预测 artifact：`runs/yolo-20260806-135408/artifacts/detection-yolo-20260806-153902.predictions.json`

正式 test 结果：

| 指标 | 值 |
| --- | ---: |
| mAP@0.5 | 0.7305 |
| mAP@0.5:0.95 | 0.3897 |
| Precision | 0.7195 |
| Recall | 0.7061 |

逐类结果：

| 类别 | Precision | Recall |
| --- | ---: | ---: |
| `hand` | 0.9013 | 0.7786 |
| `scope_control_body` | 0.7022 | 0.6706 |
| `scope_mid_section` | 0.5549 | 0.6692 |

备注：训练曲线中第 1 个 epoch 的 val mAP50 最高，后续没有继续提升。后续分析认为主要与数据量、强增强、预训练权重已经较强、训练分布和验证分布不完全匹配有关。

### 2026-08-07：AdamW 稳定训练尝试

目标：降低初始学习率，引入 AdamW、warmup、loss 权重、较保守增强，验证是否能超过 strong-cos。

配置来源：

- `runs/optimized_configs/yolo11s-group1-large-stable-adamw.yaml`
- resolved 配置：`runs/yolo-20260806-154457/config.resolved.json`

核心训练参数：

| 参数 | 值 |
| --- | --- |
| weights | `yolo11s.pt` |
| imgsz | `640` |
| epochs | `40` |
| batch | `16` |
| optimizer | `AdamW` |
| lr0 / lrf | `0.0015 / 0.05` |
| weight_decay | `0.01` |
| warmup_epochs | `5` |
| box / cls / dfl | `7.5 / 0.7 / 1.5` |
| mosaic / mixup | `0.7 / 0.05` |

产物：

- 训练目录：`runs/yolo-20260806-154457/`
- 正式评测：`runs/yolo-20260806-154457/evals/detection-yolo-20260807-155210.evaluation.json`
- 预测 artifact：`runs/yolo-20260806-154457/artifacts/detection-yolo-20260807-155210.predictions.json`

正式 test 结果：

| 指标 | 值 |
| --- | ---: |
| mAP@0.5 | 0.6854 |
| mAP@0.5:0.95 | 0.3707 |
| Precision | 0.7317 |
| Recall | 0.6808 |

结论：precision 略高，但 mAP 和 recall 不如 strong-cos；不作为最佳模型。注意该目录 `status.json` 仍显示 `running`，但已存在完整 evaluation JSON，属于状态文件未及时更新的遗留问题。

### 2026-08-07：从 strong-cos best 继续低学习率 768 微调

目标：从当前 best 权重继续微调，尝试 `imgsz=768`、低学习率、弱增强，观察是否改善小目标和贴近目标。

配置来源：

- `runs/optimized_configs/yolo11s-group1-large-ft768-lowlr-lowaug.yaml`
- resolved 配置：`runs/yolo11s-ft768-lowlr-lowaug/config.resolved.json`

核心训练参数：

| 参数 | 值 |
| --- | --- |
| weights | `runs/yolo-20260806-135408/.../best.pt` |
| imgsz | `768` |
| epochs | `24` |
| batch | `8` |
| optimizer | `SGD` |
| lr0 / lrf | `0.001 / 0.2` |
| warmup_epochs | `2` |
| rect | `true` |
| mosaic / mixup | `0.0 / 0.0` |

产物：

- 训练目录：`runs/yolo11s-ft768-lowlr-lowaug/`
- 最佳权重：`runs/yolo11s-ft768-lowlr-lowaug/checkpoints/group1_large_yolo11s_ft768_lowlr_lowaug_frombest/weights/best.pt`
- 训练曲线：`runs/yolo11s-ft768-lowlr-lowaug/checkpoints/group1_large_yolo11s_ft768_lowlr_lowaug_frombest/results.png`

结论：训练成功，但训练曲线最佳 val mAP50 约 `0.6054`，低于 strong-cos；未作为最终检测模型。

### 2026-08-10 至 2026-08-11：YOLO 单帧后处理全量对比

目标：不重新训练模型，只在保存的预测结果上比较后处理策略。

脚本：

- `benchmark/cli/postprocess_detection.py`

产物：

- 总报告：`runs/postprocess_reports/yolo11s-strong-cos-best-postprocess.md`
- JSON：`runs/postprocess_reports/yolo11s-strong-cos-best-postprocess.json`
- split CSV：`runs/postprocess_reports/yolo11s-strong-cos-best-postprocess-splits.csv`
- tracker 初步报告：`runs/postprocess_reports/yolo11s-strong-cos-best-trackers.md`

关键结果：

| 策略 | mAP50 | mAP50-95 | Precision | Recall | F1 | Predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 逐类阈值 + top-k | 0.6169 | 0.3296 | 0.8167 | 0.7421 | 0.7776 | 18797 |
| 逐类阈值 + NMS@0.55 | 0.6193 | 0.3307 | 0.8050 | 0.7454 | 0.7741 | 19153 |
| 全局 conf=0.30 + NMS@0.55 | 0.6508 | 0.3416 | 0.7524 | 0.7745 | 0.7633 | 21294 |

结论：固定 top-k 的离线 F1 高，但不适合真实视频，因为路人经过时可能出现 4 只手等动态目标数；更建议使用不固定目标数的阈值 + NMS/Soft-NMS 策略。

### 2026-08-13：检测可视化工具补强

目标：用 `F:\暑期实习\test` 中视频快速查看 YOLO 框效果。

代码：

- 修改 `tools/visualize_detections.py`
- 增加 `max_frames`，可以只渲染片段预览。

产物：

- `runs/visualizations/test_yolo11s_best_conf030_preview20s.mp4`

### 2026-08-14：接入 Label Studio 原生轨迹 GT，重做 tracker 评测

目标：原 YOLO txt 标签只有逐帧框，没有 track id；改用 Label Studio 导出的原生 `videorectangle.sequence`，每个检测物时间轴作为一个 GT track。

数据：

- Label Studio 导出：`datasets/labelstudio-yolo-test/project-15-at-2026-08-13-05-16-2be8e556.json`
- YOLO test 帧：`datasets/cleansight-yolo/group1_large/images/test`
- 对应方式：Label Studio `task.id` 对应图片名前缀 `t{task_id}_xxxxxx.jpg`
- 成功对应 task 数：41
- 成功对应帧数：4290

脚本：

- `tools/evaluate_labelstudio_trackers.py`
- `tools/render_tracking_comparison.py`

产物：

- 报告：`runs/labelstudio_track_eval/labelstudio_gt_track_eval.md`
- JSON：`runs/labelstudio_track_eval/labelstudio_gt_track_eval.json`
- track 前视频：`runs/labelstudio_track_eval/task97_before_track_predict_conf0.30.mp4`
- track 后视频：`runs/labelstudio_track_eval/task97_after_track_botsort_conf0.30_iou0.70.mp4`

关键结果：

| 方法 | Precision | Recall | F1 | IDF1 | ID Switches | Fragments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `botsort_conf0.30_iou0.70` | 0.6352 | 0.7504 | 0.6881 | 0.4791 | 289 | 295 |
| `tracktrack_conf0.30_iou0.70` | 0.6658 | 0.6060 | 0.6345 | 0.4699 | 82 | 237 |
| `ocsort_conf0.30_iou0.70` | 0.5984 | 0.7089 | 0.6490 | 0.4669 | 240 | 328 |

结论：BoT-SORT 默认配置综合最好；TrackTrack ID switch 少但 recall 损失太大，不适合作为动作时序分割前端。

### 2026-08-18：tracker 控制变量实验

目标：重点提升检测框连续性，兼顾准确率；固定 best YOLO，只改 tracker 参数。

报告：

- `runs/labelstudio_tracker_controlled/CONTROLLED_TRACKER_REPORT.md`
- 配置目录：`runs/labelstudio_tracker_controlled/configs/`

最佳配置：

- `runs/labelstudio_tracker_controlled/configs/botsort_high_clean.yaml`

推荐参数：

```yaml
tracker_type: botsort
track_high_thresh: 0.35
track_low_thresh: 0.15
new_track_thresh: 0.35
track_buffer: 30
match_thresh: 0.8
fuse_score: true
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: false
model: auto
```

关键结果：

| 方法 | Precision | Recall | F1 | IDF1 | ID Switches | Fragments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BoT-SORT high-clean | 0.6461 | 0.7473 | 0.6930 | 0.5204 | 255 | 294 |
| BoT-SORT match=0.90 | 0.6243 | 0.7593 | 0.6852 | 0.5084 | 262 | 311 |
| BoT-SORT default | 0.6352 | 0.7504 | 0.6881 | 0.4791 | 289 | 295 |

可视化：

- 三栏对比视频：`runs/labelstudio_tracker_controlled/task95_predict_default_optimized_botsort_trails.mp4`

结论：更保守的轨迹启动/维持阈值最有效；单纯加大 `track_buffer` 没有改善；放宽关联阈值会造成错关联。

### 2026-08-21：val 选参的后处理探索与 tracker 输入参数复验

目标：避免直接在 test 上搜索最优参数；用 val 选后处理，再在 test 上验证，同时比较 `imgsz=640/768`。

脚本：

- `tools/explore_yolo_postprocess_next.py`

产物：

- 总结：`runs/yolo_postprocess_next/YOLO_POSTPROCESS_NEXT_SUMMARY.md`
- 预测 artifact：`runs/yolo_postprocess_next/artifacts/*.predictions.json`

关键结果：

| 输入尺寸 | Val 最优策略 | Test Precision | Test Recall | Test F1 | Test mAP50 | Test mAP50-95 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 640 | `conf=0.30 + Gaussian Soft-NMS@0.55` | 0.7591 | 0.7729 | 0.7659 | 0.6507 | 0.3413 |
| 768 | `conf=0.30 + Gaussian Soft-NMS@0.55` | 0.7534 | 0.7169 | 0.7347 | 0.6111 | 0.3269 |

结论：`imgsz=768` 没有改善，当前检测后处理推荐 `imgsz=640 + conf=0.30 + Gaussian Soft-NMS@0.55`。

同日进一步做了 tracker 输入参数近似复验：

- 报告：`runs/labelstudio_tracker_postprocess_next/THREE_EXPERIMENTS_NEXT_ANALYSIS.md`
- 新实验结果：`runs/labelstudio_tracker_postprocess_next/labelstudio_gt_track_eval.md`
- track 前视频：`runs/labelstudio_tracker_postprocess_next/task97_before_track_predict_conf0.25.mp4`
- track 后视频：`runs/labelstudio_tracker_postprocess_next/task97_after_track_botsort_high_clean_conf0.25_iou0.55.mp4`

关键结果：

| 方法 | Precision | Recall | F1 | IDF1 | ID Switches | Fragments |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `conf=0.25, iou=0.55` | 0.6562 | 0.7522 | 0.7009 | 0.5416 | 199 | 285 |
| `conf=0.40, iou=0.55` | 0.6726 | 0.7257 | 0.6981 | 0.5316 | 167 | 322 |
| `conf=0.30, iou=0.55` | 0.6595 | 0.7464 | 0.7003 | 0.5157 | 195 | 294 |
| `conf=0.30, iou=0.70` | 0.6458 | 0.7487 | 0.6935 | 0.4984 | 259 | 294 |

结论：视频离线推理主方案更推荐 `YOLO conf=0.25 + NMS IoU=0.55 + BoT-SORT high-clean`。它比旧的 `conf=0.30 + iou=0.70` IDF1 提升 `+0.0432`，ID Switches 减少 60 次。

### 2026-08-24：GPU 延迟测试

目标：分别测试 ROI 特征提取耗时和 YOLO 检测后 tracker 后处理耗时。

脚本：

- `tools/benchmark_gpu_roi_track_latency.py`

产物：

- ROI 报告：`runs/gpu_latency_bench/ROI_FEATURE_LATENCY_REPORT.md`
- Track 报告：`runs/gpu_latency_bench/TRACK_POSTPROCESS_LATENCY_REPORT.md`
- JSON：`runs/gpu_latency_bench/gpu_roi_track_latency_report.json`

测试环境：

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存总量 | 8188 MiB |
| PyTorch | 2.11.0+cu128 |
| TorchVision | 0.26.0+cu128 |
| Ultralytics | 8.4.115 |

ROI 特征提取结果：

| 指标 | 值 |
| --- | ---: |
| 图片帧数 | 6786 |
| ROI 总数 | 20686 |
| Backbone | ResNet50 |
| ROI size | 224 x 224 |
| ROI batch | 64 |
| 总耗时 | 159.0011 s |
| 平均每 ROI 总耗时 | 7.6864 ms |
| GPU forward 吞吐 | 784.9140 ROI/s |
| torch peak allocated | 404.2363 MiB |
| torch peak reserved | 592.0000 MiB |

Tracker 后处理结果：

| 指标 | 值 |
| --- | ---: |
| 图片帧数 | 6786 |
| YOLO 检测框总数 | 22320 |
| 平均每帧检测框 | 3.2891 |
| Track 后处理总耗时 | 148.4261 s |
| Track 平均每帧 | 21.8724 ms |
| Track 吞吐 | 45.7197 FPS |
| YOLO 检测预计算耗时 | 151.9700 s |
| 包含检测预计算总 wall time | 303.2651 s |

结论：当前 tracker 本身约 45.7 FPS，可用于离线视频；端到端瓶颈主要来自 YOLO 检测、图片读取和 OpenCV/GMC 等 CPU 流程。

### 2026-08-25：ROI backbone 对比

目标：为后续动作模型接入前评估不同 ROI backbone 的速度、显存和 tracker 外观特征效果。

脚本：

- `tools/compare_roi_backbones_for_tracking.py`

产物：

- 总报告：`runs/roi_backbone_tracker_compare/ROI_BACKBONE_TRACKER_COMPARE_SUMMARY.md`
- JSON：`runs/roi_backbone_tracker_compare/roi_backbone_tracker_compare.json`
- 单独报告：`runs/roi_backbone_tracker_compare/*_report.md`

固定条件：

| 项目 | 值 |
| --- | --- |
| YOLO 权重 | `runs/yolo-20260806-135408/.../best.pt` |
| Label Studio GT task | 41 |
| 评测帧数 | 4290 |
| YOLO batch | 16 |
| ROI batch | 64 |
| imgsz | 640 |
| conf / NMS IoU / max_det | 0.25 / 0.55 / 20 |
| Tracker | BoT-SORT high-clean |
| 精度 | fp16 |

综合结果：

| Backbone | ROI/s | ms/ROI | Peak Alloc MiB | F1 | IDF1 | ID Switches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no-ReID baseline | - | - | - | 0.6167 | 0.4755 | 363 |
| resnet18 | 1092.08 | 0.9157 | 311.08 | 0.6168 | 0.4759 | 373 |
| mobilenet_v3_small | 1052.60 | 0.9500 | 133.06 | 0.6172 | 0.4772 | 371 |
| efficientnet_b0 | 712.20 | 1.4041 | 378.69 | 0.6164 | 0.4804 | 364 |
| convnext_tiny | 492.89 | 2.0289 | 477.83 | 0.6171 | 0.4806 | 393 |
| dinov2_vits14 | 689.95 | 1.4494 | 228.98 | 0.6167 | 0.4783 | 384 |

结论：通用预训练 ROI embedding 对 tracker IDF1 提升很小，`convnext_tiny` IDF1 最高但 ID Switches 增加，`mobilenet_v3_small` 显存最低，`resnet18` 速度最快。若 ROI 特征用于动作模型，不应只看 tracker 指标，还需要用动作分类/时序分割指标复验。

### 2026-08-28：交接整理

本次新增本文档，并删除专门 smoke 文件和临时 smoke 目录。当前还没有创建新的 git commit。

## 4. 当前推荐方案

检测模型仍推荐使用：

```text
runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt
```

视频离线推理 V1 推荐：

```yaml
detection:
  model: yolo11s_strong_cos_best
  imgsz: 640
  conf: 0.25
  nms_iou: 0.55
  max_det: 20

tracker:
  type: botsort
  track_high_thresh: 0.35
  track_low_thresh: 0.15
  new_track_thresh: 0.35
  track_buffer: 30
  match_thresh: 0.8
  fuse_score: true
  gmc_method: sparseOptFlow
  with_reid: false
```

如果只做单帧检测、不接 tracker，推荐：

```yaml
imgsz: 640
conf: 0.30
postprocess:
  type: gaussian_soft_nms
  iou: 0.55
  sigma: 0.5
```

## 5. 重要注意事项

1. YOLO txt 标签仍是逐帧检测格式，不包含 track id；轨迹 GT 需要从 Label Studio 原生导出恢复。
2. `runs/` 目录被 `.gitignore` 忽略，所有实验报告、视频和权重都不会随 git commit 上传。
3. `runs/yolo-20260806-154457/status.json` 状态显示 `running`，但它已有正式 evaluation JSON；不要只根据该状态文件判断是否还在训练。
4. tracker 不能解决检测器长期漏检，也不能把“两只手合成一个框”的检测结果拆成两个实例；这类问题需要补数据、改检测模型或尝试实例分割。
5. 当前通用 ROI backbone 接入 tracker 没有显著提升 IDF1，后续若用于动作模型，应以动作模型指标为准。

## 6. 后续建议

1. 把本轮未提交代码做一次 commit，建议 commit message：`feat(yolo): add postprocess tracker and ROI benchmark tools`。
2. 若需要远端复现，至少同步 `runs/optimized_configs/*.yaml`、新增工具脚本和本文档；权重、数据、视频、报告需要走网盘或对象存储。
3. 下一步实验优先级：
   - 实现“自定义 Soft-NMS 输出 -> tracker”的离线管线，重新评估 IDF1。
   - 在全部 41 个 Label Studio task 上复验 `conf=0.25 + nms_iou=0.55 + BoT-SORT high-clean`。
   - 对 1 到 3 帧短断轨做线性/Kalman 插值，观察 Fragments 和 IDF1 是否改善。
   - 针对控制端漏检和手部贴近合并补充训练样本，必要时评估实例分割模型。
