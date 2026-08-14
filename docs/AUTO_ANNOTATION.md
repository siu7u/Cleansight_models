# YOLO 自动标注工具（auto-annotate）

用已训练 YOLO checkpoint 对无标注视频逐帧检测，自动产出**与历史 Label Studio 导出同构**的
标注 JSON（videorectangle 轨迹），作为时序模型的训练输入（检测特征侧）。

## 数据流

```
新视频 (mp4)
  │  python -m framework.cleansight_eval.cli.annotate \
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
python -m framework.cleansight_eval.cli.annotate \
    --videos path/to/video.mp4 --config framework/experiments/auto-annotate.yaml

# 目录内全部视频
python -m framework.cleansight_eval.cli.annotate \
    --videos path/to/videos/ --config framework/experiments/auto-annotate.yaml

# 覆盖阈值 / 输出目录 / smoke 探针
python -m framework.cleansight_eval.cli.annotate --videos ... --config ... \
    --conf 0.3 --out outputs/annotations_smoke --max-frames 30

# 帧采样加速（每 4 帧推理一次，中间帧沿用最近结果，推理成本降 4 倍）
python -m framework.cleansight_eval.cli.annotate --videos ... --config ... \
    --frame-stride 4

# ByteTrack 实例跟踪（轨迹按实例 id 组织，帧间实例连续）
python -m framework.cleansight_eval.cli.annotate --videos ... --config ... \
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
  建议抽样与人工标注对照（检出率 / IoU）。

## 验证

```bash
# 单元测试（含 legacy lab.py 兼容性验收，不依赖 ultralytics）
pytest framework/tests/test_auto_annotate.py -v

# 全链路 smoke（需要 ultralytics 与 legacy 权重）
pytest framework/tests/test_auto_annotate.py -v -k smoke
```
