# 内镜清洗 时序动作识别 流程文档（YOLO → MS-TCN++）

> 给后续同学开发/复现用的参考。本文覆盖从原始标注到「逐帧动作识别 + 可视化」的**完整链路**，
> 包含每个脚本的作用、数据契约、超参、目录约定，以及踩过的坑。
>
> 配套物料（已随包提供）：
> - `endoscope/`            —— 本流程的胶水脚本 + YOLO 基座 + 标注 JSON + YOLO 训练输出
> - `MS-TCN2/`              —— MS-TCN++ 框架（含 `data/Endo_Project` 数据、`models/.../epoch-50.model`、可视化脚本）
> - `weights/best.pt`       —— **生产用 YOLO 权重**（87.7MB，对应 `my_yolo_small_object_opt`）

---

## 0. 一句话概括

> 一段内镜清洗视频 → **YOLO 逐帧检测 4 类物体**（手/长刷头/镜口/短刷）→ 把每帧检测压成 **20 维特征向量** →
> **MS-TCN++ 在时间维上做时序分类**，输出**逐帧动作标签**（Idle / Long_Brushing / Short_Brushing）→ 评估 + 叠加可视化。

YOLO 负责「这一帧画面里有什么、在哪」（空间），MS-TCN 负责「这段时间在做什么动作」（时序）。两者通过 **20 维/帧的特征序列**解耦。

---

## 1. 两套类别体系（最容易搞混，先记住）

| 层 | 模型 | 类别 | 数量 | 来源 |
|----|------|------|------|------|
| **空间检测** | YOLOv8 | `Hand` `Long_Brush_Head` `Scope_Port` `Short_Brush` | **4 类物体** | Label Studio 框标注 |
| **时序识别** | MS-TCN++ | `Idle` `Long_Brushing` `Short_Brushing` | **3 类动作** | Label Studio 时间轴标注 |

- YOLO 的类别顺序固定：`{0: Hand, 1: Long_Brush_Head, 2: Scope_Port, 3: Short_Brush}`（见 `extract_features.py` `CLASS_MAP`，**必须与训练时 data.yaml 一致**）。
- MS-TCN 的类别顺序由 `MS-TCN2/data/Endo_Project/mapping.txt` 决定（按字母排序自动生成）：
  ```
  0 Idle
  1 Long_Brushing
  2 Short_Brushing
  ```

---

## 2. 全链路总图

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ① 标注 (Label Studio)                                                       │
│    project-7-at-2026-05-01-...json                                          │
│      ├─ 帧级框标注  ──────────────► YOLO 训练数据                            │
│      └─ 时间轴标注 (timelinelabels) ─► MS-TCN 逐帧动作标签                   │
└──────────────────────────────────────────────────────────────────────────┘
        │                                              │
        ▼ (YOLO 分支)                                  ▼ (MS-TCN 标签分支)
┌─────────────────────────────┐          ┌─────────────────────────────────┐
│ ② YOLO 数据集准备            │          │ ⑤ MS-TCN 标签准备                 │
│   split_dataset.py          │          │   prepare_mstcn_data.py           │
│   merge_yolo_datasets.py    │          │   JSON 时间轴 → 逐帧 groundTruth   │
│   → split-yolo-data/        │          │   + 自动生成 mapping.txt           │
│     (images/labels + yaml)  │          │   (rescale 对齐 + 动作>Idle 优先级)│
└──────────────┬──────────────┘          └─────────────────┬─────────────────┘
               ▼                                            │
┌─────────────────────────────┐                            │
│ ③ YOLO 训练/验证             │                            │
│   train_yolo8.py (yolov8n)  │                            │
│   test_yolo8.py / predict_… │                            │
│   → best.pt                 │                            │
└──────────────┬──────────────┘                            │
               ▼                                            │
┌─────────────────────────────┐                            │
│ ④ 特征提取 (YOLO→特征)       │                            │
│   extract_features.py       │                            │
│   每帧取每类最高分检测        │                            │
│   → (N, 20) .npy            │                            │
│   preprocess_features.py    │                            │
│   转置 → (20, N)            │                            │
└──────────────┬──────────────┘                            │
               └───────────────┬────────────────────────────┘
                               ▼
                ┌──────────────────────────────────────┐
                │ ⑥ MS-TCN++ 训练 / 预测 / 评估          │
                │   MS-TCN2/main.py  --action train     │
                │   MS-TCN2/main.py  --action predict   │
                │   MS-TCN2/eval.py                     │
                │   → results/.../<vid> 逐帧动作         │
                │   visualize_full.py 叠加框+动作标签视频 │
                └──────────────────────────────────────┘
```

---

## 3. 逐阶段详解

### ① 标注 —— `project-7-at-2026-05-01-15-54-631a0db0.json`
Label Studio 导出，一个 JSON 里同时含两类标注：
- **框标注**：每帧物体 bbox（YOLO 用）。
- **时间轴标注 `timelinelabels`**：`ranges: [{start, end}]` + 动作 label（MS-TCN 用）。`start/end` 是**原始视频帧号**。

### ② YOLO 数据集准备 —— `split_dataset.py` / `merge_yolo_datasets.py`
- `merge_yolo_datasets.py dataset1 dataset2 output`：合并两个 YOLO 数据集（处理重名图片）。
- `split_dataset.py input output --train 0.8 --val 0.1 --test 0.1`：按 8:1:1 划分，生成标准 YOLO 目录结构（`images/{train,val,test}` + `labels/...`）并**自动写 `data.yaml`**（从 `classes.txt` 解析类名）。
- 产物：`split-yolo-data/`（含 `data.yaml`）。

### ③ YOLO 训练 / 验证 —— `train_yolo8.py` / `test_yolo8.py` / `predict_single.py`
- 训练：`yolov8n.pt` 基座 finetune，`epochs=100, imgsz=640, batch=32, patience=10, amp=False`，输出到 `runs/yolo8_finetune/`（曲线、混淆矩阵、`weights/best.pt` 都在 `runs/` 里，已随包提供作训练效果参考）。
- 评估：`test_yolo8.py` 在 test split 上跑 `model.val()`，打印 mAP50 / mAP50-95 / P / R。
- 单图预测：`predict_single.py` 对单张图推理并打印各目标类别+置信度。
- **⚠️ 注意权重有两个版本**：
  - `runs/yolo8_finetune/weights/best.pt` —— 4/12 的早期 finetune（脚本里默认指向它）。
  - **`weights/best.pt`（= `my_yolo_small_object_opt/weights/best.pt`，4/26）—— 这才是 dev 指定的生产权重**，`extract_features.py` 里的 `MODEL_PATH` 用的是它。复现请以这个为准。

### ④ 特征提取 —— `extract_features.py` + `preprocess_features.py`（关键）
**这是 YOLO 与 MS-TCN 的接口层。**

`extract_features.py`：对每个视频**逐帧** YOLO 推理（`conf=0.25`），把每帧压成 **20 维向量**：

```
20 维 = 4 类物体 × 5 个值
每类的 5 个值 = [x_center, y_center, width, height, conf]   # 坐标用 xywhn 归一化(0~1)，与分辨率无关
排布：  [ Hand(0:5) | Long_Brush_Head(5:10) | Scope_Port(10:15) | Short_Brush(15:20) ]
```

- **同一类多个目标 → 只保留置信度最高的那个**（`best_detections`）。⚠️ 这是个简化，多实例场景会丢信息。
- **某类未检出 → 该段 5 个值全 0**（`conf==0` 即代表"本帧没有这个物体"，可视化脚本据此判断是否画框）。
- 输出：每视频一个 `.npy`，形状 **(N, 20)**，N=总帧数。

`preprocess_features.py`：把 (N, 20) **转置成 (20, N)**，以适配 MS-TCN 的 `Conv1d`（通道在前、时间在后）。
- ✅ 本包 `MS-TCN2/data/Endo_Project/features/*.npy` 已经是 **(20, N)** 格式（转置已应用），可直接训练/预测。

### ⑤ MS-TCN 标签准备 —— `prepare_mstcn_data.py`
把 Label Studio 时间轴标注转成**与特征逐帧对齐**的 groundTruth：
1. 默认全帧填 `Idle`。
2. **Rescale**：标注的 `start/end` 是原始帧号，特征帧数可能不同 → 用 `scale_factor = 特征长度 / 标注最大帧号` 把区间映射到特征时间轴。
3. **优先级覆盖**：`Long_Brushing / Short_Brushing (prio=2) > Idle (prio=1)`，处理标注重叠。
4. 输出每视频一个 `.txt`（每行一个动作标签，行数 = 特征帧数），并**自动生成 `mapping.txt`**（标签按字母排序编号）。

### ⑥ MS-TCN++ 训练 / 预测 / 评估 —— `MS-TCN2/`
开发者给的命令（`features_dim=20` 必须显式传，默认是 2048）：

```bash
cd MS-TCN2
# 训练
python main.py --action=train   --dataset=Endo_Project --split=1 --num_epochs=50 \
               --num_layers_PG=10 --num_layers_R=10 --num_R=3 --features_dim=20
# 预测（加载 models/Endo_Project/split_1/epoch-50.model）
python main.py --action=predict --dataset=Endo_Project --split=1 --num_epochs=50 \
               --num_layers_PG=10 --num_layers_R=10 --num_R=3 --features_dim=20
# 评估
python eval.py --dataset=Endo_Project --split=1
```

- **预测产物**：`MS-TCN2/results/Endo_Project/split_1/<vid>`，首行 `### Frame level recognition: ###`，第二行空格分隔的逐帧动作标签。
- **评估指标**（`eval.py`）：帧准确率 Acc、分段编辑距离 Edit、F1@{0.1, 0.25, 0.5}（按段 IoU 匹配）。

### 叠加可视化 —— `MS-TCN2/visualize_full.py`
把三样东西画到原视频上输出 mp4：YOLO 框（按 20 维特征还原像素坐标）、真值动作（绿）、预测动作（预测对=绿/错=红）。
- 已自带 (20,N)↔(N,20) 自适应转置。产物示例：`MS-TCN2/results/vis/export17_final_overlay.mp4`。

---

## 4. MS-TCN++ 模型结构（`MS-TCN2/model.py`）

```
输入 (B, 20, T)
  └─► Prediction Generation (PG)         num_layers_PG=10
        双分支膨胀卷积(dilation 正序+逆序) 融合 + 残差, num_f_maps=64
        → 初版逐帧 logits (B, 3, T)
  └─► Refinement × num_R(=3)             每个 num_layers_R=10, dilation=2^i
        对上一级 softmax 结果再卷积细化
  输出 outputs (num_R+1=4, B, 3, T)       取最后一级 [-1] 作为最终预测
```

- **Loss** = 交叉熵 + `0.15 × 截断平滑MSE`（相邻帧 log-softmax 的 MSE，抑制抖动、鼓励时序平滑）。
- 关键超参（务必和上面命令一致）：`features_dim=20`、`num_classes=3`、`num_f_maps=64`、`num_layers_PG=num_layers_R=10`、`num_R=3`、`lr=5e-4`、`bz=1`、`sample_rate=1`。
- 已训练权重：`MS-TCN2/models/Endo_Project/split_1/epoch-50.model`（3.38MB，state_dict）。推理只需这一个 `.model`；`.opt`（优化器状态）仅续训用，本包未含。

---

## 5. 目录与数据约定

```
MS-TCN2/data/Endo_Project/
├── features/            # (20, N) .npy，每视频一个  ← 来自 ④
├── groundTruth/         # 逐帧动作 .txt，每视频一个 ← 来自 ⑤（注意是 groundTruth 驼峰）
├── splits/
│   ├── train.split1.bundle   # 16 个视频
│   └── test.split1.bundle    # 4 个视频（export16-480P / export17 / export1 / export12）
└── mapping.txt          # 0 Idle / 1 Long_Brushing / 2 Short_Brushing
```

`batch_gen.py` 读数据时用 `common_length = min(特征帧数, 标签行数)` 强制对齐（两者常差 1 帧，正常）。

---

## 6. 踩坑 / 注意事项（复现必看）

1. **`features_dim` 必须传 20**：`main.py` 默认 2048（原论文 I3D 特征维度），不传会维度不匹配直接报错。
2. **特征方向 (N,20) vs (20,N)**：`extract_features.py` 出来是 (N,20)，MS-TCN 的 Conv1d 要 (20,N)，**必须先过 `preprocess_features.py` 转置**。本包 `data/features` 已转好。
3. **groundTruth 目录命名**：`prepare_mstcn_data.py` 里 `OUTPUT_PATH` 写的是 `ground_truth`（下划线），但 MS-TCN 的 `main.py` 读的是 `groundTruth`（驼峰）。生成后**需重命名/移动到 `groundTruth/`**（本包已是 `groundTruth/`）。
4. **特征目录跨项目**：`extract_features.py` 默认存到 `endoscope/data/...`，而 MS-TCN 从 `MS-TCN2/data/...` 读。复现时记得把特征/标签放进 `MS-TCN2/data/Endo_Project/` 下。
5. **YOLO 权重选型**：用 `weights/best.pt`（生产版，4/26），不是 `runs/yolo8_finetune/weights/best.pt`（早期版）。
6. **每类只取最高分检测**：多个同类目标会被合并成一个，是当前特征设计的已知局限。
7. **`main.py` 的 shebang 写的是 python2.7** 但代码用了 `loguru` 等，实际按 **Python3** 跑。
8. **eval 的背景类**：`eval.py` 默认 `bg_class=["background"]`，而本项目没有 `background`，所以 **`Idle` 会被当作正常动作段参与 Edit/F1 统计**（不是背景），解读指标时注意。

---

## 7. 端到端复现顺序（TL;DR）

```
标注(JSON)
  ├─ YOLO: split_dataset → train_yolo8 → best.pt
  │         └─ extract_features(用 best.pt) → preprocess_features → MS-TCN2/data/.../features (20,N)
  └─ 标签: prepare_mstcn_data → 重命名为 groundTruth/ → MS-TCN2/data/.../groundTruth
                                                              │
                       MS-TCN2: main.py train → predict → eval.py → visualize_full.py
```

依赖（两端都要）：`torch`、`ultralytics`（YOLOv8）、`opencv-python`、`numpy`、`pandas`、`tqdm`、`loguru`。
