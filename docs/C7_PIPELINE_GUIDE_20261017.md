# C7 生产推理管线使用指南（2026-10-17）

## 快速开始

```bash
# 输入：视频 + YOLO 检测框目录 + 视频名
.venv/bin/python tools/c7_pipeline.py \
  --video <视频.mp4> \
  --frames-dir <检测框目录> \
  --stem <视频名.mp4> \
  --out <输出.json> [--labels <标签.txt>]
```

输出 JSON 含 `segments`（v1.2 后处理段列表 [start, end, label]）、`segments_raw`、
`decode_config`、模型信息；给 `--labels` 时附带与 GT 的 edit/F1 对照。

## 前置条件

| 项 | 说明 |
|---|---|
| C7 交付包 | `role-hands-c7-mstcn2-v41/`（84MB 权重，gitignore 单独管理，勿删） |
| YOLO 检测框 | auto-annotate 产物（`framework/experiments/auto-annotate-yolo11.yaml`，stride 2） |
| DINOv2 | 首次联网下载（已缓存则离线）；嵌入按视频缓存于 `tmp/c7_hand_embeddings/`（重跑秒级） |

## 权重选择（关键）

| 场景 | --weights | --hand-state | 说明 |
|---|---|---|---|
| **新视频（生产默认）** | all（20 个平均） | all | README 建议的临时方案；**真实性能未经评测**（见下） |
| 复现评测 | fold<k>（4 个） | fold<k> | 视频所在折的 held-out 权重 |

### ⚠️ 为什么不能用 all-20 评测已有视频

他们的 5 折在全部 43 视频上划分——all-20 里对任一已有视频有 16/20 权重**训练时见过它**。
实测对比（test 8 视频，all-20 vs fold 专用）：均值 71.1 vs 66.4（+4.7）——**这是数据泄漏的
虚高，不是真实增益**。诚实数字 = fold 专用：test edit 中位 71.8 / 均值 66.4。

**正式上线前必须在全部 43 视频上重训**（README §8 有完整命令，TODO #4，需 GPU）。

## 已知限制（详见 docs/C7_ROLE_HANDS_EVALUATION_20261017.md）

- 仅离线（角色平滑用未来 0.5s + 双向卷积 + v1.2 中心窗平滑）
- 纯 flush 视频可能幻觉长毛刷动作（67edf4f9 案例，edit 23-31）
- flush 帧级 F1 < 0.1（syringe 检出率 3% 的上游瓶颈）