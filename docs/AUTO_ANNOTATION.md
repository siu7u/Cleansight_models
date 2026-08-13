# YOLO 自动标注工具（auto-annotate）

用已训练 YOLO checkpoint 对无标注视频逐帧检测，自动产出**与历史 Label Studio 导出同构**的
标注 JSON（videorectangle 轨迹），作为时序模型的训练输入（检测特征侧）。

## 数据流

```
新视频 (mp4)
  │  python -m framework.cleansight_eval.cli.annotate \
  │      --videos <视频或目录> --config framework/experiments/auto-annotate.yaml
  ▼
outputs/auto_annotations/<视频名>.json   ← legacy 标注 JSON（每视频一个）
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
    --conf 0.3 --out outputs/auto_annotations_smoke --max-frames 30
```

`--runs-dir` 控制 ultralytics 中间产物目录（默认 `outputs/ultralytics_runs`，
仓库内且被 Git 忽略）；ultralytics 8.3 默认可能把中间产物写到按安装位置推断的
git 仓库根（如 `CleanSightBackend/runs`），本工具统一重定向，避免污染其他项目。

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
| 轨迹划分 | 每类别按框面积取 top-K：`hand` 2 条、其他 1 条（`top_k` 可配置） | YOLO 无实例跟踪；与 `clean_bbox_v2` 的 slot 语义一致 |
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
- **实例不关联**：同一类别多目标按每帧面积排序取 top-K，帧间不跟踪实例；轨迹序号
  语义与 `clean_bbox_v2` 的 slot 一致（slot0=最大框）。
- **质量门**：自动标注质量取决于 YOLO checkpoint 与置信度阈值；大规模并入训练集前
  建议抽样与人工标注对照（检出率 / IoU）。

## 验证

```bash
# 单元测试（含 legacy lab.py 兼容性验收，不依赖 ultralytics）
pytest framework/tests/test_auto_annotate.py -v

# 全链路 smoke（需要 ultralytics 与 legacy 权重）
pytest framework/tests/test_auto_annotate.py -v -k smoke
```
