# 推理链路性能测量与"预计算 vs 现场推理"决策（2026-08-17）

## 背景与决策问题

自动标注通道的训练特征来自 YOLO 推理。要决定：

- **方案 A（预计算，当前设计）**：YOLO 推理结果一次性固化进数据集（`annotate run` →
  `convert`），之后每次训练直接读缓存特征。
- **方案 B（现场推理）**：每次训练运行时对视频现场跑 YOLO 推理生成特征。

本文档用真实数据与生产代码路径实测全链路时延，支撑该决策。

## 测量环境与方法

- 环境：CPU（CleanSightBackend venv，torch 2.8.0，无 GPU）
- 口径：30fps 视频、`frame_stride=4`（等效 7.5fps 采样）、imgsz=640、conf=0.25、batch=16
- 复用生产代码：`auto_annotate.run.detect_video`（YOLO 检测，legacy large+small 双
  checkpoint）、`temporal.data.load_split / featurize_frame_bbox`（特征化）、
  `temporal.models` + 修复后 checkpoint（时序推理）
- 视频样本：短 15.5s / 中 139.6s / 长 277.4s 各一个真实视频

## 各环节实测时延

| 环节 | 实测 | 说明 |
|---|---|---:|---|
| YOLO 检测（stride 4） | **0.077-0.079s / 秒视频**（CPU 2m36s / GPU 2m33s） | 2026-08-18 实测：空闲 CPU 全量 17 视频 156s；RTX 4060 Laptop 全量 153s——两者持平，**瓶颈在 CPU 侧解码/预处理（cv2 解码 ~5-8ms/帧），GPU 推理被掩盖**；首轮测量 0.64-0.75s/s 是后台训练抢 CPU 的虚高值；检测近似确定（结构一致，conf 微抖 <0.001） |
| 特征化（bbox → 40 维） | 0.16 ms/帧（含 IO）；纯计算 0.022 ms/帧 | 可忽略 |
| 训练加载缓存特征（load_split） | 7 视频 7216 帧共 1.16s | 每 epoch 只付这一笔 |
| MS-TCN 全序列推理（T=2081） | 3.1 ms/视频 | 毫秒级，非瓶颈 |
| Transformer 全序列推理（T=2081） | 37.5 ms/视频 | 毫秒级，非瓶颈 |
| GRU 滑窗单 tick（window=16） | 1.49 ms | 7.5fps 帧预算 133ms 的 1% |

## 决策测算（17 个视频全量 ≈ 1987.6s 视频时长）

| 路径 | 成本 | 测算 |
|---|---|---:|
| 方案 A 预计算（一次性） | YOLO 检测 + convert | **2.6 分钟（CPU，实测）**，每数据集版本只付一次 |
| 方案 B 现场推理（每 run 一次） | 每次训练重跑 YOLO | **2.6 分钟/run**，N 次训练 = N × 2.6 分钟 |
| 方案 B 现场推理（每 epoch 重推理） | 30 epoch 全部重跑 | ≈ **78 分钟/run**（1.3 小时） |

> 2026-08-18 实测两次：空闲 CPU `time annotate run` 全量 17 视频 = **real 2m36.5s**；RTX 4060 Laptop（`torch.cuda.is_available()=True`）= **real 2m33.1s**。GPU 未带来提速：瓶颈在 CPU 侧视频解码与预处理（14907 推理帧 ≈ 97 帧/s 吞吐，GPU 纯推理理论 ~30s，其余 ~120s 是解码/预处理/后处理）。首轮估算（13-25 分钟）作废：当时测量与后台训练并发，CPU 争抢使速率虚高约 9 倍。

训练后每 epoch 的固定开销（方案 A）：读缓存特征 1.16s，比方案 B 的推理成本低 3-4 个数量级。

## 结论与建议

1. **采用方案 A（预计算固化到数据集）**：一次性成本 **2.6 分钟（CPU/GPU 实测持平）**，
   训练同一数据版本 ≥2 次即回本；若训练会调参多次（本周的 3 模型对比就是
   同数据跑 3 次以上），节省按 run 数线性放大。**当前设计（`annotate run` → `convert`）
   是正确选择，无需改为现场推理。**
2. **现场推理的合理场景**只有两个：数据量小到 YOLO 成本可忽略（几秒级视频），或
   特征来源本身是实验变量（如做 YOLO 权重/阈值消融时避免反复 convert）——即便如此也
   建议先 YOLO 推理缓存为中间 JSON，再按需特征化。
3. **推理侧（部署）不是瓶颈**：全序列毫秒级、滑窗单 tick 1.49ms 远低于 7.5fps 帧预算；
   YOLO 检测若要提速，瓶颈不在 GPU 而在解码/预处理——可考虑并行解码、更大 batch，
   或 GPU 侧解码（NVDEC，需更换读帧实现）。
4. **测量口径注意**：GPU 实测（RTX 4060 Laptop）与 CPU 持平（2m33s vs 2m36s），原因是管线瓶颈在 CPU 解码/预处理；若换更大 batch/GPU 解码可进一步压缩，但决策不变（预计算仍更优）。

## 复现

```bash
# 测量脚本（临时，不入 Git）：tmp/bench_inference_chain.py + tmp/bench_seq_timing.py
/home/caizh/programming/python_code/CleanSightBackend/.venv/bin/python tmp/bench_inference_chain.py
```
