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
|---|---:|---|
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

## 图片数据集链（run-dataset）实测（2026-08-20）

预标注场景（图片帧序列 + 动作标签 → YOLO 自动检测 → frames/ 入数据集）的推理速度实测：

- 环境：RTX 4060（torch 2.13.0+cu130）/ CPU（同机），legacy large+small 双 checkpoint，
  imgsz=640、conf=0.25、batch=16、预热 1 batch 后取稳态，脚本 `tmp/bench_annotate_images.py`
- 样本：`datasets/cleansight-ActionMixed/images/train` 真实图片帧 400 张

| 环节 | GPU | CPU |
|---|---:|---:|
| 全链路（imread + 批量推理） | **98.6 帧/s**（10.1 ms/帧） | **11.5 帧/s**（87 ms/帧） |
| 纯推理（_infer_batch） | 122.3 帧/s（8.2 ms/帧） | 11.6 帧/s（86.5 ms/帧） |
| 纯解码（cv2.imread） | 1146 帧/s（0.9 ms/帧） | 1134 帧/s |

- **与视频链的关键差异**：图片链没有视频容器解码，解码开销可忽略（0.9 ms/帧），
  瓶颈在 YOLO 推理本身 → **GPU 有 ~8.5 倍加速**（98.6 vs 11.5 帧/s），不像视频链
  被 CPU 解码卡死（GPU/CPU 持平）。
- 全量实测：`run-dataset --dataset datasets/cleansight-ActionMixed`（9532 个标签帧，
  train/val/test 全 split）GPU 后台完成约 2 分钟内（按稳态外推 ~97s）；
  CPU 外推约 14 分钟。
- 预标注训练流程已实测打通（2026-08-20）：run-dataset 产物 → `cli.train` MS-TCN 3
  epoch smoke（GPU 0.17s/epoch，val_acc 26.75）→ `benchmark.cli.eval` 出
  acc/edit/macro-F1 报告，全链路 exit 0。

### 图片链决策

与视频链一致，**采用方案 A（预计算，YOLO 结果上传进数据集）**：全量 9532 帧一次性
成本 GPU ~1.6 分钟（CPU ~14 分钟），训练时每 epoch 只付 `load_split` 读缓存 ~1.2s；
现场推理则每次训练 run 都要重付。图片链的预计算产物即 `frames/`（run-dataset 默认
原地写入数据集根，`--out` 可重定向），决策不随设备变化（GPU/CPU 只是成本快慢差异）。

## 时序训练过程：直接读数据集缓存，无实时推理

**结论：时序训练过程中没有任何实时 YOLO 推理，检测特征全部来自训练前固化进数据集的缓存文件。**

```
训练前（一次性、离线）：run / run-dataset → YOLO 推理 → 结果固化进数据集（frames/ + labels/）
训练中（每个 epoch）：   cli.train → temporal.data.load_split → 只读缓存 txt → 40 维特征 → 模型
                        （全程无 ultralytics / YOLO 调用）
```

- 代码证据：训练走 `temporal/data.py::load_split`，只读 `frames/<split>/<序列>-<帧号>.txt`
  （训练前写好的 bbox 缓存）+ `labels/<split>/`（动作标签），训练管线不 import ultralytics。
- 成本证据：每 epoch 只付一次缓存读取，实测 7 视频 7216 帧共 **1.16s**（比现场推理低
  3-4 个数量级）。
- 方案对比（全量 9532 帧，图片链）：

| 路径 | 成本 | 说明 |
|---|---|---:|
| 方案 A 预计算（当前实现） | 一次性 GPU ≈ 1.4 分钟 + 每 epoch 读缓存 ~1.2s | 同一数据版本特征固定、可复现 |
| 方案 B 训练时现场推理 | 每次训练/每 epoch 重付 1.4 分钟（GPU） | 30 epoch ≈ 42 分钟/run |
| 回本点 | 训练同一数据 **≥2 次** 即方案 A 更优 | 三模型对比 = 同数据 3+ 次 |

- 部署侧澄清：CleanSightBackend 生产预测时是现场跑 YOLO → 特征化 → 时序模型
  （实时发生在部署服务里）；"实时推理"只存在于**训练前标注**与**部署预测**两个环节，
  **不在训练循环内**。

### 用户环境复测（2026-08-21，汇报用双环境对照）

同一脚本、同一权重/样本，用户环境（RTX 4060）复测结果：

| 环节 | 参考环境（08-20） | 用户环境（08-21） | 说明 |
|---|---|---:|---:|
| GPU 全链路 | 98.6 帧/s（10.1 ms/帧） | **111.9 帧/s**（8.9 ms/帧） | 达标且更快 |
| CPU 全链路 | 11.5 帧/s（87 ms/帧） | **13.9 帧/s**（72 ms/帧） | 达标 |
| 检测量（200 帧，双 checkpoint 合并） | 1090 框 | 1089 框 | 检测结果一致（NMS 微小抖动） |

换算（用户环境）：全量 9532 帧 → GPU **85 秒 ≈ 1.4 分钟**、CPU **686 秒 ≈ 11.4 分钟**，
预计算决策不变。

## 复现

```bash
# 图片链测速（临时脚本，不入 Git）：tmp/bench_annotate_images.py
# 依赖环境：pip install --target=/home/caizh/cs-annotate-env torch torchvision ultralytics lap pytest numpy pyyaml opencv-python-headless
PYTHONPATH=/home/caizh/cs-annotate-env python3 tmp/bench_annotate_images.py 400       # GPU
CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/caizh/cs-annotate-env python3 tmp/bench_annotate_images.py 400  # CPU

# 图片链全量预标注 + 训练 + 评测（2026-08-20 实测路径）
python -m framework.cleansight_eval.cli.annotate run-dataset \
    --dataset datasets/cleansight-ActionMixed --config framework/experiments/auto-annotate.yaml \
    --out outputs/actionmixed-auto-images
python -m framework.cleansight_eval.cli.train \
    --config tmp/exp-images-auto-smoke.yaml --runs-dir outputs/runs_annotate_images_smoke
python -m benchmark.cli.eval \
    --config tmp/exp-images-auto-smoke.yaml \
    --ckpt outputs/runs_annotate_images_smoke/mstcn-*/checkpoints/best.pt

# 测量脚本（临时，不入 Git）：tmp/bench_inference_chain.py + tmp/bench_seq_timing.py
/home/caizh/programming/python_code/CleanSightBackend/.venv/bin/python tmp/bench_inference_chain.py
```
