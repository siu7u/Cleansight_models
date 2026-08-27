# 自动标注模块快速上手（auto-annotate）

用已训练 YOLO checkpoint 给无标注数据自动生成检测标注。两条链共用同一份配置：

- **视频链**：视频 → legacy 标注 JSON →（合并人工动作标签）→ 时序训练数据
- **数据集链**：图片帧序列（+ 动作标签）→ 时序训练数据（frames/ + labels/）

完整说明见 [`AUTO_ANNOTATION.md`](AUTO_ANNOTATION.md)，配置字段见 `usage/YAML_CONFIG.md` §8。
本文只给最小可跑的命令集。

## 0. 前置条件

- 在**仓库根目录**、你的 venv 中执行所有命令
- 依赖：`ultralytics`、`opencv-python`、`pyyaml`、`lap`（`--track` 的 ByteTrack
  需要 lap，smoke 测试也会用到）、`pytest`（验证用）
- 权重：`legacy/yolo-detection/pipeline/versioned_weights/` 下的
  `yolo-large-v3/best.pt`（hand / scope_control_body / scope_mid_section）与
  `yolo-small-v3/best.pt`（syringe / air_gun / scope_distal_end）
- 配置：`framework/experiments/auto-annotate.yaml`（`run` 用 checkpoints / imgsz /
  conf / top_k / frame_stride / batch_size / track；`run-dataset` 用 checkpoints /
  imgsz / conf / batch_size）

## 1. 视频自动标注（run + convert）

```bash
# ① 自动标注：单视频
python -m framework.cleansight_eval.cli.annotate run \
    --videos path/to/video.mp4 --config framework/experiments/auto-annotate.yaml

#    批量：整个视频目录 + 帧采样加速 + 断点续跑（推荐）
python -m framework.cleansight_eval.cli.annotate run \
    --videos path/to/videos/ --config framework/experiments/auto-annotate.yaml \
    --frame-stride 4 --resume
```

产出：`outputs/annotations/<视频名>.json`（与 Label Studio 导出同构）。
**注意：run 只产出检测框，动作标签必须来自人工 Label Studio 导出。**

```bash
# ② 检查质量（可选但建议）：画框预览
python tools/visualize_annotations.py \
    --json outputs/annotations/<视频名>.json \
    --video path/to/<视频名>.mp4 --output outputs/visualizations/<视频名>_preview.mp4

# ③ 转换：自动检测 + 人工动作标签 → 时序训练数据（train/val 各跑一次）
python -m framework.cleansight_eval.cli.annotate convert \
    --annotations outputs/annotations \
    --labels-export legacy/yolo-detection/pipeline/raw/exports/<人工导出>.json \
    --out datasets/cleansight-ActionMixed-auto --split train
python -m framework.cleansight_eval.cli.annotate convert \
    --annotations outputs/annotations \
    --labels-export legacy/yolo-detection/pipeline/raw/exports/<人工导出>.json \
    --out datasets/cleansight-ActionMixed-auto --split val
```

产出：`datasets/<数据集>/labels/<split>/<视频>.mp4.txt`（帧号 + 动作 id，7.5fps 抽样）+
`frames/<split>/<视频>.mp4-<帧号:06d>.txt`（bbox 行）+ 类别映射 `data.yaml`。

## 2. 数据集自动标注（run-dataset：图片帧序列 → 时序训练数据）

```bash
# 基本用法：数据集根（images/<split>/<序列>-<帧号:06d>.jpg 有序帧 + labels/<split>/<序列>.txt 动作标签）
# 默认原地补写 frames/，产出与 convert 同构，temporal/data.py 直接消费
python -m framework.cleansight_eval.cli.annotate run-dataset \
    --dataset datasets/xxx --config framework/experiments/auto-annotate.yaml

# 输出到独立目录 / 断点续跑 / 覆盖阈值
python -m framework.cleansight_eval.cli.annotate run-dataset \
    --dataset datasets/xxx --config framework/experiments/auto-annotate.yaml \
    --out datasets/xxx-auto --resume --conf 0.3
```

产出：`frames/<split>/<序列>-<帧号:06d>.txt`（逐帧 bbox `class_id cx cy w h`，
8 类全局编号，仅覆盖有动作标签的帧）+ 动作标签原样复制到输出根 labels/ +
`labels/data.yaml`（6 类动作）+ `frames/data.yaml`（8 类检测，缺省补写）。
之后可直接训练时序模型（smoke 全链路见 §5 B4–B6）。

## 3. 常用参数速查

| 参数 | 子命令 | 作用 |
|---|---|---|
| `--videos` | run | 视频文件或目录（缺省取配置 `videos`） |
| `--frame-stride N` | run | 每 N 帧推理一次，中间帧沿用最近结果（成本降 N 倍，30fps 建议 4） |
| `--track` | run | 启用 ByteTrack 实例跟踪（轨迹按实例 id 组织） |
| `--max-frames N` | run | 每视频最多推理帧数（smoke 探针） |
| `--dataset` | run-dataset | 数据集根目录（`images/<split>/<序列>-<帧号:06d>.jpg` 有序帧 + `labels/<split>/<序列>.txt` 动作标签） |
| `--out` | run-dataset | 时序训练数据输出根（默认数据集根原地补写 `frames/`） |
| `--conf X` | 两者 | 覆盖全局置信度阈值（配置可写 `{类别: 阈值}` 类别级） |
| `--imgsz N` / `--batch-size N` | 两者 | 推理输入尺寸 / 批量大小 |
| `--resume` | 两者 | 跳过已存在产出的视频/图片（断点续跑） |
| `--annotations` / `--labels-export` / `--out` / `--split` | convert | 标注 JSON 目录 / 人工导出 / 输出根 / split 名 |

## 4. 验证

```bash
# 单元测试（期望 30 passed，不依赖 ultralytics）
python -m pytest framework/tests/test_auto_annotate.py -q

# 全链路 smoke（期望 3 passed，需要 ultralytics + legacy 权重 + lap）
python -m pytest framework/tests/test_auto_annotate.py -v -k smoke
```

常见报错排查：

| 报错 | 原因与处理 |
|---|---|
| `配置缺少 checkpoints 列表` | 配置里没有 checkpoints（每项含 path + class_map） |
| `class_map 含权重中不存在的类别 id` | 本地 id 与权重 names 不一致，修正配置 |
| `数据集缺少 images/ 目录` / `缺少 labels/ 目录` | run-dataset 输入根必须含 images/（有序帧）与 labels/（动作标签） |
| `序列 <seq> 缺少动作标签` | 图片有但 labels/<split>/<seq>.txt 缺失，时序训练必需动作标签 |
| `标签帧 <f> 在 images/ 中无对应图片` | labels 引用的帧号在图片里不存在，检查命名/帧号范围 |
| `图片文件名缺少 '-<帧号:06d>' 后缀` | 图片必须命名为 `<序列>-<帧号:06d>.<ext>`（如 demo.mp4-000141.jpg） |

## 5. 开箱即跑（本仓库真实路径，全部在仓库根、你的 venv 中执行）

> 需要装有 ultralytics + opencv-python + pyyaml + lap 的环境；推理命令依赖
> `legacy/yolo-detection/pipeline/versioned_weights/{yolo-large-v3,yolo-small-v3}/best.pt`
> 与配置 `framework/experiments/auto-annotate.yaml`（均已就位）。
> 本节的 B 系列命令已实测跑通（2026-08-20）；A 系列为视频链（历史已落地）。

**视频链（run → convert → 训练）**：

```bash
# A1 自动标注 smoke：单视频前 60 帧，输出到 scratch（不碰已有的 outputs/annotations）
python -m framework.cleansight_eval.cli.annotate run \
    --videos legacy/yolo-detection/pipeline/raw/videos/05ba4406-clip_1781584018103_1781584033616.mp4 \
    --config framework/experiments/auto-annotate.yaml \
    --out outputs/annotations_smoke --max-frames 60

# A2 画框预览，人工检查检测质量
python tools/visualize_annotations.py \
    --json outputs/annotations_smoke/05ba4406-clip_1781584018103_1781584033616.json \
    --video legacy/yolo-detection/pipeline/raw/videos/05ba4406-clip_1781584018103_1781584033616.mp4 \
    --output outputs/visualizations/smoke_preview.mp4

# A3 convert smoke：自动检测 + 人工动作标签 → 训练数据（scratch 输出）
python -m framework.cleansight_eval.cli.annotate convert \
    --annotations outputs/annotations_smoke \
    --labels-export legacy/yolo-detection/pipeline/raw/exports/project-10-at-2026-07-07-19-32.json \
    --out outputs/smoke_convert --split train

# A4 全量重跑：17 个视频，帧采样 + 断点续跑（已有产物自动跳过，幂等安全）
python -m framework.cleansight_eval.cli.annotate run \
    --videos legacy/yolo-detection/pipeline/raw/videos/ \
    --config framework/experiments/auto-annotate.yaml \
    --frame-stride 4 --resume

# A5 时序训练 + 评测（smoke 配置 10 epoch；数据已是现成的 datasets/cleansight-ActionMixed-auto）
python -m framework.cleansight_eval.cli.train \
    --config framework/experiments/mstcn-autoannotate-smoke.yaml
python -m benchmark.cli.eval \
    --config framework/experiments/mstcn-autoannotate-smoke.yaml \
    --ckpt runs/autoannotate-smoke/mstcn-*/checkpoints/best.pt
```

**数据集链（run-dataset → 时序训练）**：

```bash
# B1 从现成时序数据集抽 1 个序列的前 10 个标签帧，造 smoke 数据集
SRC=datasets/cleansight-ActionMixed
SEQ=$(basename "$(ls $SRC/labels/train/*.txt | head -1)" .txt)   # 序列名（含 .mp4）
mkdir -p outputs/ds_smoke/images/train outputs/ds_smoke/labels/train
head -10 $SRC/labels/train/$SEQ.txt > outputs/ds_smoke/labels/train/$SEQ.txt
for f in $(cut -d' ' -f1 outputs/ds_smoke/labels/train/$SEQ.txt); do
  cp $SRC/images/train/$SEQ-$(printf %06d $f).jpg outputs/ds_smoke/images/train/
done

# B2 自动标注：图片帧序列 + 动作标签 → 时序训练数据（默认原地补写 frames/）
python -m framework.cleansight_eval.cli.annotate run-dataset \
    --dataset outputs/ds_smoke \
    --config framework/experiments/auto-annotate.yaml
# 期望: [auto-annotate] train/<SEQ>: 10 帧 → frames/train/   然后 exit=0

# B3 检查产物（frames 行格式与 convert 一致：class_id cx cy w h）
cat outputs/ds_smoke/frames/train/$SEQ-*.txt | head
ls outputs/ds_smoke/frames/ outputs/ds_smoke/labels/           # data.yaml + train

# B4 消费验收：产物可被 temporal/data.py 直接读出 40 维特征
python -c "
from cleansight_eval.temporal import data as td
feats, truths, id2name = td.load_split({'root': 'outputs/ds_smoke', 'labels_dir': 'labels', 'frames_dir': 'frames'}, split='train')
print('序列数:', len(feats), '| 特征形状:', feats[0].shape)      # 期望 (10, 40)
print('标签:', truths[0].tolist())
print('动作类别:', list(id2name.values()))
"

# B5 训练 smoke（2 epoch；split_val 指向 train 只因 smoke 集只有 train，只验证链路）
sed -e 's|outputs/actionmixed-auto-images|outputs/ds_smoke|' \
    -e 's|epochs: 3|epochs: 2|' -e 's|split_val: val|split_val: train|' \
    tmp/exp-images-auto-smoke.yaml > tmp/exp-ds-smoke.yaml
python -m framework.cleansight_eval.cli.train \
    --config tmp/exp-ds-smoke.yaml --runs-dir outputs/runs_ds_smoke
# 期望: cat outputs/runs_ds_smoke/mstcn-*/status.json → "state": "succeeded"

# B6 评测
python -m benchmark.cli.eval \
    --config tmp/exp-ds-smoke.yaml --ckpt outputs/runs_ds_smoke/mstcn-*/checkpoints/best.pt
# 期望: 输出 evaluation.json / best.eval.md

# B7 真实数据集全量：cleansight-ActionMixed 是现成"图片帧 + 动作标签"时序数据集
#    （9532 帧 GPU ≈2 分钟；--out 指向 scratch，避免改动原数据集）
python -m framework.cleansight_eval.cli.annotate run-dataset \
    --dataset datasets/cleansight-ActionMixed \
    --config framework/experiments/auto-annotate.yaml \
    --out outputs/actionmixed-auto-images
```

> 注意：当前两个 legacy 权重只覆盖 6 类（hand / scope_control_body /
> scope_mid_section / syringe / air_gun / scope_distal_end），`short_brush` /
> `brush_tip_out` 检测不到（对应帧为空文件），这是配置现状；只对有动作标签的
> 帧产出 frames。B 链 smoke 产物在 `outputs/`（Git 忽略），不会污染仓库。

## 6. 相关文档

- 完整使用指南：[`AUTO_ANNOTATION.md`](AUTO_ANNOTATION.md)（数据流 / 输出格式 / 设计决策 / 正式训练结果）
- 人工审核闭环：[`YOLO_REVIEW_FLOW.md`](YOLO_REVIEW_FLOW.md)（预标注 → Label Studio 人工改框 → convert）
- 配置说明：[`usage/YAML_CONFIG.md`](../usage/YAML_CONFIG.md) §8
- 代码入口：`framework/cleansight_eval/cli/annotate.py` + `framework/cleansight_eval/detection/auto_annotate/`
