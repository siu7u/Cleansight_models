# YOLO 自动标注工具（auto-annotate）

用已训练 YOLO checkpoint 对无标注视频逐帧检测，自动产出**与历史 Label Studio 导出同构**的
标注 JSON（videorectangle 轨迹），作为时序模型的训练输入（检测特征侧）。

> **定位：自动标注是新增数据通道，不替代手动标注训练。**
> 手动标注数据集 `temporal.actionmixed-v2`（人工标注检测框 + 人工动作标签）及其训练
> 能力原样保留：登记、split manifest、训练配置（`*-actionmixed.yaml`）与评测链路均不受
> 本工具影响。自动标注数据另立数据集条目 `temporal.actionmixed-auto-v1`，两条通道并存，
> 手动标注是正式 benchmark 的锚点，自动标注用于扩量与对照（量化自动标注特征代价）。

## 数据流

```
新视频 (mp4)
  │  python -m framework.cleansight_eval.cli.annotate run \
  │      --videos <视频或目录> --config framework/experiments/auto-annotate.yaml
  ▼
outputs/annotations/<视频名>.json   ← legacy 标注 JSON（每视频一个）
  │  可被历史代码直接消费：
  │    legacy/temporal-transformer/lab.py::load_data_json  → [T, N*5] 特征
  │    legacy/yolo-detection/pipeline/utils/lsexport.py    → YOLO 训练集
  ▼
动作标签（timelinelabels）由人工 Label Studio timeline 标注补充
  ▼
时序模型训练（feature = YOLO 检测标注，label = 人工动作标签）
```

## 用法

```bash
# 单视频
python -m framework.cleansight_eval.cli.annotate run \
    --videos path/to/video.mp4 --config framework/experiments/auto-annotate.yaml

# 目录内全部视频
python -m framework.cleansight_eval.cli.annotate run \
    --videos path/to/videos/ --config framework/experiments/auto-annotate.yaml

# 覆盖阈值 / 输出目录 / smoke 探针
python -m framework.cleansight_eval.cli.annotate run --videos ... --config ... \
    --conf 0.3 --out outputs/annotations_smoke --max-frames 30

# 帧采样加速（每 4 帧推理一次，中间帧沿用最近结果，推理成本降 4 倍）
python -m framework.cleansight_eval.cli.annotate run --videos ... --config ... \
    --frame-stride 4

# ByteTrack 实例跟踪（轨迹按实例 id 组织，帧间实例连续）
python -m framework.cleansight_eval.cli.annotate run --videos ... --config ... \
    --track

# 断点续跑（跳过已存在产出的视频）
python -m framework.cleansight_eval.cli.annotate run --videos ... --config ... \
    --resume

# convert：标注 JSON + 人工动作标签 → 时序训练数据布局
# （检测来自自动标注，动作标签来自人工 Label Studio timelinelabels，
#  帧号自动做 LS 帧率 → 真实帧率换算）
python -m framework.cleansight_eval.cli.annotate convert \
    --annotations outputs/annotations \
    --labels-export legacy/yolo-detection/pipeline/raw/exports/project-10-at-2026-07-07-19-32.json \
    --out datasets/cleansight-ActionMixed-auto --split train
```

convert 产出（与 `temporal/data.py` 消费契约一致）：
- `labels/<split>/<video>.mp4.txt`：抽样帧 `"frame_id action_id"`（1-based 真实帧号，~7.5fps）
- `frames/<split>/<video>.mp4-<f:06d>.txt`：逐帧 bbox（5 列，兼容 40 维 v1 特征）
- `labels/data.yaml` + `frames/data.yaml`：类别映射（6 类动作 / 8 类检测）

转换后的数据可直接用 `framework/experiments/mstcn-autoannotate-smoke.yaml` 训练
（smoke 验证全链路：自动标注 → 训练 → 评测，实测 10 epoch 跑通）。

`--runs-dir` 控制 ultralytics 中间产物目录（默认 `outputs/ultralytics_runs`，
仓库内且被 Git 忽略）；ultralytics 8.3 默认可能把中间产物写到按安装位置推断的
git 仓库根（如 `CleanSightBackend/runs`），本工具统一重定向，避免污染其他项目。

优化参数（均可写入配置 YAML）：

| 参数 | 说明 |
|---|---|
| `frame_stride` | 每 N 帧推理一次，中间帧沿用最近推理结果；30fps 视频建议 4（等效 7.5fps，与训练采样率一致）。实测 120 帧 CPU 推理 10.1s → 1.6s。 |
| `batch_size` | 批量推理帧数（默认 16），GPU 利用率更高。 |
| `track` | 启用 ByteTrack 实例跟踪，轨迹按 `(类别, 实例 id)` 组织，替代 top-K 伪轨迹。 |
| `conf` | 标量或 `{类别: 阈值}` 字典（按最低阈值推理、逐类过滤，小目标类别可放宽）。 |
| `--resume` | 跳过已存在产出的视频，批量中断后可续跑。 |

配置说明见 `framework/experiments/auto-annotate.yaml`（登记于 `usage/YAML_CONFIG.md` §8）。

## 检查检测结果（人工检查 YOLO 检测质量）

`annotate run` 产出 JSON 后、进入 convert/训练前，建议先人工检查检测质量
（大规模并入训练集前的质量门，见「已知限制」）。

### 数字层面：JSON 检查

```bash
python -c "
import json
j = json.load(open('outputs/annotations/<视频名>.json'))
for ann in j[0]['annotations']:
    for r in ann['result']:
        v = r['value']; seq = v['sequence']
        en = [e for e in seq if e.get('enabled')]
        confs = [e.get('conf', 0) for e in en]
        print(f\"{v['labels'][0]}: enabled {len(en)}/{len(seq)}, conf {min(confs):.2f}-{max(confs):.2f}\")"
```

关注点：每条轨迹的 enabled 覆盖率（应接近 100%，异常低说明该类别漏检严重）、
conf 分布（过低说明检测不稳）。

### 直观层面：画框预览视频

```bash
# 把检测框画回视频帧，输出预览视频（框上带类别名 + 置信度）
python tools/visualize_annotations.py \
    --json outputs/annotations/<视频名>.json \
    --video legacy/yolo-detection/pipeline/raw/videos/<视频名>.mp4 \
    --output outputs/visualizations/<视频名>_preview.mp4

# 只画高置信度框 / 只处理前 300 帧（长视频快速预览）
python tools/visualize_annotations.py --json ... --video ... --output ... \
    --conf 0.4 --max-frames 300
```

判据：框是否贴合目标、有无明显误检/漏检、类别是否正确。不合格时：
- 调 `conf` 阈值或类别级阈值（`auto-annotate.yaml`）
- 补覆盖缺失类别的 YOLO 权重
- 或走人工修正闭环（`docs/YOLO_REVIEW_FLOW.md`）

## 完整工作流（视频 → 时序模型训练）

```bash
# ① 自动标注：目录内全部视频（帧采样加速 + 断点续跑）
python -m framework.cleansight_eval.cli.annotate run \
    --videos legacy/yolo-detection/pipeline/raw/videos/ \
    --config framework/experiments/auto-annotate.yaml \
    --frame-stride 4 --resume

# ② 转换：自动检测 + 人工动作标签 → 训练数据（train/val 各跑一次）
python -m framework.cleansight_eval.cli.annotate convert \
    --annotations outputs/annotations \
    --labels-export legacy/yolo-detection/pipeline/raw/exports/project-10-at-2026-07-07-19-32.json \
    --out datasets/cleansight-ActionMixed-auto --split train
python -m framework.cleansight_eval.cli.annotate convert \
    --annotations outputs/annotations \
    --labels-export legacy/yolo-detection/pipeline/raw/exports/project-10-at-2026-07-07-19-32.json \
    --out datasets/cleansight-ActionMixed-auto --split val

# ③ 训练（smoke 配置；正式训练请扩展数据并登记 testsets.yaml）
python -m framework.cleansight_eval.cli.train \
    --config framework/experiments/mstcn-autoannotate-smoke.yaml

# ④ 评测
python -m benchmark.cli.eval \
    --config framework/experiments/mstcn-autoannotate-smoke.yaml \
    --ckpt runs/autoannotate-smoke/mstcn-*/checkpoints/best.pt
```

### 正式训练（数据扩量并登记 testsets.yaml 后）

数据集扩到多个视频后，把自动标注数据登记进 `framework/testsets.yaml`
（新增 `temporal.actionmixed-auto-v1` 数据集条目 + train/val/test manifest，
`split_overlap_policy: frame`，`validate_catalog` 要求同一 dataset_version 必须
同时登记 train 和 test），再用正式配置训练。同一 auto 数据上三模型对比：

```bash
# 三模型正式训练（MS-TCN 全序列 / GRU 滑窗 / Transformer 全序列）
python -m framework.cleansight_eval.cli.train \
    --config framework/experiments/mstcn-actionmixed-auto.yaml
python -m framework.cleansight_eval.cli.train \
    --config framework/experiments/gru-actionmixed-auto.yaml
python -m framework.cleansight_eval.cli.train \
    --config framework/experiments/transformer-actionmixed-auto.yaml

# 各自评测（best.pt 路径以训练输出为准）
python -m benchmark.cli.eval \
    --config framework/experiments/mstcn-actionmixed-auto.yaml \
    --ckpt runs/<run>/mstcn-*/checkpoints/best.pt
```

注意：auto 数据最长视频（a2ade960，277.4s @30fps）按 7.5fps 抽样后序列约 2080 帧，
transformer 配置的 `max_len` 已设为 2560；GRU 的 `window: 16` 需 ≤ 最短视频采样帧数
（auto 数据最短约 35 帧）。对照基线可用历史人工标注数据集 `temporal.actionmixed-v2`
（`python -m framework.cleansight_eval.cli.dataset --preset actionmixed` 下载）跑同模型，
量化自动标注特征 vs 人工标注特征的代价。

> 注意：① 只产出检测标注；动作标签必须来自人工标注（② 的 `--labels-export`
> 提供 timelinelabels）。convert 会把同一视频在人工导出中的动作标签与自动检测合并，
> 并自动做 LS 标注帧率 → 视频真实帧率的帧号换算。

## 输出格式（对齐 legacy）

单个 JSON 为 task 数组，结构与 `legacy/yolo-detection/pipeline/raw/exports/*.json`
的 videorectangle 逐字段一致：

```jsonc
[{
  "id": 0,
  "data": {"video": "xxx.mp4"},
  "annotations": [{"result": [
    {
      "type": "videorectangle",
      "value": {
        "labels": ["hand"],
        "framesCount": 510,
        "duration": 21.25,
        "sequence": [
          {"frame": 1, "enabled": true, "rotation": 0,
           "x": 40.0, "y": 35.0, "width": 20.0, "height": 10.0,
           "time": 0.0, "conf": 0.91}
        ]
      }
    }
  ]}]
}]
```

## 设计决策

| 决策 | 约定 | 理由 |
|---|---|---|
| 轨迹划分 | 默认每类别按框面积取 top-K（`hand` 2 条、其他 1 条）；`track=true` 时按 ByteTrack 实例 id 组织真实轨迹 | top-K 与 `clean_bbox_v2` 的 slot 语义一致；跟踪后帧间实例连续，轨迹更真实 |
| sequence 密度 | 全帧写入，覆盖 `[1, framesCount]` | legacy `interpolate_sequence` 要求轨迹等长，全帧无损 |
| 缺席帧 | `enabled=false` + 外推上一有效框坐标（从未出现则全 0） | 与人工标注"离场点"语义一致；`lsexport.build_segments` 正确截断 |
| 坐标 | YOLO 归一化中心点 → 左上角百分比，裁剪到 [0,100] | 与 Label Studio 导出同口径 |
| 置信度 | sequence 帧附加非标准 `conf` 字段 | legacy 解析器忽略未知字段，完全兼容；保留真实置信度供新框架将来使用 |
| 帧号 | 从 1 开始，`time=(frame-1)/fps`，`duration=framesCount/fps` | 与人工标注一致 |
| 动作标签 | 不产出 timelinelabels | YOLO 无法生成动作标签，由人工 timeline 标注补充 |

## 已知限制

- **类别覆盖**：示例配置使用 legacy 权重 `yolo-large-v3`（hand / scope_control_body /
  scope_mid_section）与 `yolo-small-v3`（syringe / air_gun / scope_distal_end），共 6 类。
  `short_brush`、`brush_tip_out`（时序 40 维特征的第 7/8 类）和 `long_brush`（CLEAN
  recipe 目标）目前没有对应 legacy 权重；需要时在 `checkpoints` 中追加覆盖这些类别的
  模型（代码由配置驱动，无需改动）。
- **实例关联**：未启用 `track` 时同一类别多目标按每帧面积排序取 top-K，帧间不跟踪
  实例（slot0=最大框，与 `clean_bbox_v2` 一致）；启用 `track` 后由 ByteTrack 维护
  实例 id，目标消失重现会换新 id（轨迹拆段，符合 legacy 多轨迹语义）。
- **质量门**：自动标注质量取决于 YOLO checkpoint 与置信度阈值；大规模并入训练集前
  建议抽样与人工标注对照（检出率 / IoU）。需要人工逐帧修正时走
  「YOLO 预标注 → Label Studio 审核 → 导出 → convert」闭环，见
  `docs/YOLO_REVIEW_FLOW.md`。

## 验证

```bash
# 单元测试（含 legacy lab.py 兼容性验收，不依赖 ultralytics）
pytest framework/tests/test_auto_annotate.py -v

# 全链路 smoke（需要 ultralytics 与 legacy 权重）
pytest framework/tests/test_auto_annotate.py -v -k smoke
```

## 正式训练结果（2026-08-17，三模型对比 + 同视频消融）

### 数据与流程

11 个带人工动作标签视频全部完成自动标注（`--frame-stride 4`，CPU 约 7 分钟）并转换登记：
`temporal.actionmixed-auto-v1`（train 7 视频 / 7216 标签帧，val 2 / 1193，test 2 / 98；
train 6 类全覆盖）。同一天用同一批模型与超参在手动标注数据 `temporal.actionmixed-v2`
（train 14 视频 / 5993 帧）上训练对照基线。训练与评测全部 formal 模式、CPU。

> **数据修复记录**：最初 2 个自动标注 JSON（05ba4406 / 4807dbbe）是本会话之前的 smoke
> 产物（enabled 检测只覆盖前 60 帧，`--resume` 会跳过），导致这两个视频的特征几乎全零。
> 已删除重跑（修复后全帧覆盖，train 非空帧文件占比从 ~85% 提升到 98-100%），
> 数据集 revision 更新为 `636e6372...`，以下指标均为修复后数据。

### 训练期验证集指标（best val_acc，修复后）

| 模型 | 自动标注数据（auto） | 手动标注数据（manual） |
|---|---:|---:|
| MS-TCN（全序列） | 16.18（epoch 2） | 66.91（epoch 30） |
| GRU（滑窗 16） | 20.81（epoch 4） | 65.21（epoch 6） |
| Transformer（全序列） | 16.18（epoch 1） | 63.30（epoch 30） |

### 正式评测（test split，修复后）

| 模型 | auto test（98 帧，仅 long_brush_withdraw） | manual test（1457 帧，7 视频） |
|---|---:|---:|
| | acc / edit | acc / edit / macro-F1 |
| MS-TCN | 0.0 / 0.0 | 53.12 / 72.80 / 58.04 |
| GRU | 0.0 / 0.0 | 33.36 / 46.17 / 54.79 |
| Transformer | 0.0 / 0.0 | 37.54 / 26.25 / 43.81 |

### 同视频消融：人工框 vs YOLO 框（2026-08-17，关键结果）

**实验设计**：9 个在两条通道重叠的视频，取公共帧网格（共 3724 帧），动作标签统一用
project-10 导出（单一标签源），唯一变量是 bbox 特征来源（人工标注框 vs YOLO 自动框）。
train 6 视频 / val 1（65d70028）/ test 2，三模型同超参。

| 模型 | 人工框（manual） | YOLO 框（auto） | 差距 |
|---|---:|---:|---:|
| MS-TCN | 78.51 | 57.74 | **-20.8** |
| GRU | 73.94 | 65.15 | -8.8 |
| Transformer | 65.47 | 18.48 | **-47.0** |

**结论**：在完全可控条件下（同视频、同帧、同标签），人工标注框特征显著优于 YOLO
自动框特征——这是自动标注通道当前的主要代价来源，幅度与模型有关（Transformer 最敏感，
MS-TCN 最稳）。改进方向：提升 YOLO 权重类别覆盖与置信度阈值、补 short_brush/
brush_tip_out 权重、或对自动框做质量门后再并入训练。

### 结论与局限

- **链路已打通**：自动标注 → 转换 → 登记 → 正式训练 → 正式评测全流程可复现。
- **auto test 0.0 是真实结果，不是 bug**：两个 test 视频全为 `long_brush_withdraw`
  （训练集中仅占 10%），模型全部预测成 idle / long_brush_insert。这是当前 11 个视频
  数据集的固有局限——类丰富的视频都进了 train，留给 test 的只有 3 个小视频（4.6-8.4s）。
- **两条通道的绝对值差距不能直接比较**（视频集合与训练量不同），以同视频消融为准。
- 结果登记见 `registry/temporal/auto-mstcn-v1`、`auto-gru-v1`、`auto-transformer-v1`。
