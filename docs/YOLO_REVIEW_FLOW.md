# YOLO 自动检测结果的人工审核流程（Human-in-the-Loop）

让 **YOLO 只做预标注**，人工在 Label Studio 里**修正检测框 + 标注动作时间线**，
审核通过后再进入时序训练。与纯自动链路（`AUTO_ANNOTATION.md`）互补，是自动标注的
质量门。

## 与两条既有链路的区别

| 链路 | 检测框来源 | 动作标签来源 | 文档 |
|---|---|---|---|
| 纯自动 | YOLO 推理（不审核） | 人工 timelinelabels | `docs/AUTO_ANNOTATION.md` |
| **自动预标注 + 人工审核（本文）** | **YOLO 预标注 → 人工修正** | 人工 timelinelabels | 本文档 |
| 纯手动 | 人工 videorectangle | 人工 timelinelabels | `temporal.actionmixed-v2`（历史数据集） |

三种链路的检测框进入训练前都必须转成同一布局（`labels/` + `frames/`，7.5fps 抽样、
40 维特征），消费侧（`temporal/data.py` + 两条时序流水线）完全共用。

## 流程总览

```
① YOLO 预标注 ──► ② 导入 Label Studio ──► ③ 人工审核 ──► ④ 导出 ──► ⑤ 拆分+convert ──► ⑥ 登记+训练
   annotate run      框显示为预标注       改框+标动作     修正后导出     人工框+人工标签      同现有流程
```

## ① YOLO 预标注

```bash
python -m framework.cleansight_eval.cli.annotate run \
    --videos <视频目录> --config framework/experiments/auto-annotate.yaml \
    --frame-stride 4 --resume
```

产物：`outputs/annotations/<视频名>.json`，每视频一个，结构为 Label Studio
videorectangle 兼容格式（含框坐标与置信度 `conf`），可直接作为 LS 预标注导入。

## ② 导入 Label Studio 作为预标注

目标项目：**项目 16 "action-train"**（label_config 已按"框由 YOLO 推理生成"配置）。
把 YOLO JSON 推为 LS 任务的 predictions，人工打开任务即可看到 YOLO 画的框：

```jsonc
// LS Prediction 结构（result 与 YOLO JSON 的 videorectangle 完全同构）
{
  "result": [
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
  ],
  "model_version": "yolo-auto-v1"
}
```

导入方式（二选一）：
- **API**：为每个视频创建 task 并附 prediction（`POST /api/projects/{id}/tasks`），
  具体请求格式依赖 LS 版本，需按实例的 OpenAPI 确认（`.env` 已配置
  `LS_HOST/LS_TOKEN/LS_AUTH`）。
- **Data Manager 文件导入**：把 YOLO JSON 作为带 pre-annotations 的任务文件导入。

> ⚠️ 这步是当前唯一未沉淀为仓库工具的环节。建议在你们 LS 实例上确认 API 后，
> 把导入脚本沉淀为仓库工具（例如 `tools/ls_import_yolo.py`）。

## ③ 人工审核（LS 界面操作）

1. **修正检测框**：拖动/增删 YOLO 预标注的 videorectangle 轨迹——改错框、删误检、
   补漏检（YOLO 只覆盖 6 类，`short_brush`/`brush_tip_out` 需人工补）。
2. **标注动作时间线**：timelinelabels 逐段标动作（idle/air_injection/flush/
   long_brush_insert/long_brush_withdraw/short_brush_cleaning）。YOLO 不产出动作
   标签，此环节必须人工。

## ④ 导出

LS 导出项目 JSON（或 API 拉取 tasks）。一份导出同时包含：**人工修正后的框
（videorectangle）+ 人工动作标签（timelinelabels）**。

## ⑤ 拆分成每视频一个 JSON + convert

convert 的 `--annotations` 约定每视频一个文件（与 auto-annotate 输出同构的
`[task]` 数组）。拆分 + 转换（解析兼容性已实测验证）：

```bash
# 拆分：导出 tasks → tmp/reviewed/<视频名>.json（每文件一个 [task] 数组）
python - <<'EOF'
import json, pathlib
tasks = json.load(open('ls-export-修正后.json'))
out = pathlib.Path('tmp/reviewed'); out.mkdir(exist_ok=True)
for t in tasks:
    name = pathlib.Path(t['data']['video']).name
    (out / name.replace('.mp4', '.json')).write_text(
        json.dumps([t], ensure_ascii=False))
EOF

# convert：检测框 = 人工修正框，动作标签 = 同一份导出的人工时间线
python -m framework.cleansight_eval.cli.annotate convert \
    --annotations tmp/reviewed \
    --labels-export ls-export-修正后.json \
    --out datasets/cleansight-ActionMixed-reviewed --split train
```

convert 自动处理：LS 标注帧率 → 视频真实帧率的帧号换算、7.5fps 抽样、8 类 bbox →
40 维特征布局（`[presence, cx, cy, w, h]` × 8 类）。

## ⑥ 登记 + 训练

同现有流程（见 `docs/AUTO_ANNOTATION.md`「正式训练」）：
1. 按 split 目录生成 manifest（参考 `benchmark/manifests/actionmixed-auto/`）
2. `framework/testsets.yaml` 登记新数据集条目（`split_overlap_policy: frame`，
   同一 dataset_version 必须同时登记 train 和 test）
3. `python tools/validate_testsets.py` 校验
4. 训练：`python -m framework.cleansight_eval.cli.train --config <正式配置>`
5. 评测：`python -m benchmark.cli.eval --config <配置> --ckpt <best.pt>`

## 已验证与待确认

| 环节 | 状态 |
|---|---|
| ① YOLO 预标注 | ✅ 工具就绪（11 视频已产出） |
| ③ 人工审核 | LS 标准操作 |
| ⑤ 拆分 + convert 兼容人工导出 | ✅ 实测（`parse_legacy_task` 解析人工导出正常，含帧号换算） |
| ⑥ 登记 + 训练 + 评测 | ✅ 实测（`temporal.actionmixed-auto-v1` 全链路） |
| ② LS 预标注导入 | ⚠️ 依赖 LS 版本，待确认 API 后沉淀工具 |

## 建议落地顺序

1. 拿 1-2 个视频走通 ①→⑥ 最小闭环（LS 导入用 Data Manager 手动导入即可，不必先写 API 脚本）
2. 确认 API 导入方式后，把 ② 和 ⑤ 的拆分沉淀为仓库正式工具
3. 批量审核时按视频量拆分任务、跟踪审核进度（LS 任务完成状态）
