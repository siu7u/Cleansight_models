# 数据集构建要求与操作指南(Label Studio 人工标注)

本指南面向**通过 Label Studio 构建/扩充训练数据集的队友**。标注产出将被
自动标注链路消费:`annotate convert`(视频链)把**人工动作标签**与 **YOLO 自动
检测框**合并成时序训练数据。**检测框可以全自动生成,动作标签必须人工标注**——
本指南的核心就是"怎么标动作标签才合格"。

参考样例(已落地):`datasets/cleansight-ActionMixed`(人工标注)、
`datasets/cleansight-ActionMixed-auto`(自动检测框 + 人工动作标签合并产物)。

## 1. 整体流程

```
① 上传视频到 Label Studio → ② 人工标注(框 + 动作 timeline)→ ③ 导出 JSON
        │                                                          │
        ▼                                                          ▼
  自动检测(annotate run,无需人工)                    convert(合并人工动作标签 + 自动检测框)
        │                                                          │
        └────────────── 时序训练数据(labels/ + frames/) ←──────────┘
```

## 2. 硬性要求(违反会跳过该视频或报错)

### 2.1 视频要求

| 要求 | 说明 |
|---|---|
| 格式 | mp4(H.264 常见即可,须能被 OpenCV 解码) |
| 命名 | 文件名**全局唯一**、稳定(convert 以文件名为序列名);参考现有 `05ba4406-clip_<时间戳>_<时间戳>.mp4` |
| 帧率/时长 | 不限(现有数据集 4.6s ~ 277s、30fps);标注帧率(LS 端)与视频真实帧率可不同,convert 自动换算 |
| 内容 | 一个视频 = 一个连续时间序列;不要把无关片段拼进一个视频 |

### 2.2 Label Studio 标注要求(每个视频必须满足)

1. **至少一个 videorectangle(检测框)标注** —— 即使只画一个框:
   - convert 依赖它提供的 `framesCount / duration` 推导 LS 标注帧率,缺了会**跳过该视频**
   - 注意:检测框本身**不需要认真标**(会自动被 YOLO 检测覆盖),它只是帧率锚点
2. **timelinelabels(动作标签)标注动作区间** —— 核心产出:
   - 标签名必须**精确匹配**以下 6 类(区分大小写与空格,否则 convert 报错):
     `idle` / `air_injection` / `flush` / `long_brush_insert` / `long_brush_withdraw` / `short_brush_cleaning`
   - **未标注的时间段默认为 `idle`**,所以只需标 5 个非 idle 动作区间
   - 区间覆盖整段动作;同一时刻只有一个动作(区间允许相邻,不允许语义重叠)

### 2.3 导出要求

- 导出全部已标注 task 的 JSON(文件名建议带日期,如 `project-10-at-<日期>.json`)
- 导出中每个视频的 `data.video` **文件名**必须与本地视频文件名一致(目录无所谓,
  convert 只取文件名匹配);参考现有导出
  `legacy/yolo-detection/pipeline/raw/exports/project-10-at-2026-07-07-19-32.json`
- 一个视频在导出里**至少有一个非空 annotation**(空 result 或缺少
  framesCount/duration 都会被 convert 跳过,如现有 14e6fadd 等 3 个视频)

## 3. 操作步骤(Label Studio)

1. 在 Label Studio 创建/复用视频标注项目,上传视频
2. 标注模板配置两类 result:
   - `videorectangle`(视频框标注)
   - `timelinelabels`(时间线标签,选项 = 上述 6 类动作名)
3. 逐视频标注:
   - 画任意一个框(帧率锚点)
   - 在时间线上框选动作区间,选择动作标签;未标区间即 idle
4. 全部完成后导出 JSON

## 4. 验收清单(提交前自查)

- [ ] 每个视频都画了框(有 framesCount/duration)
- [ ] 动作标签名与 6 类完全一致(无拼写/大小写错误)
- [ ] 导出 JSON 中每个视频有非空 annotation
- [ ] 视频文件名与本地一致、全局唯一
- [ ] 用 `annotate convert` 跑通且无该视频被跳过(命令见 §5)

## 5. 交付与接入

```bash
# 交付物:视频目录 + 导出 JSON(路径给对接人)
# 对接人接入(仓库根,自动检测已就绪时):
python -m framework.cleansight_eval.cli.annotate convert \
    --annotations outputs/annotations \
    --labels-export <你的导出.json> \
    --out datasets/<数据集名> --split train
# 期望输出:每个视频打印 "N 个标签帧";结尾不应出现该视频被跳过
```

## 6. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| convert 跳过某视频 | 导出中无该视频 / 无 framesCount/duration / 空标注 | 回到 Label Studio 补框或补动作标注后重新导出 |
| convert 报 KeyError(动作名) | 标签名不在 6 类内 | 检查拼写与大小写,重新导出 |
| 标签帧数偏少 | 导出后视频文件被替换(帧数变化) | 保持视频文件与标注时的版本一致 |
| 想扩训练集但不想标框 | 框是帧率锚点,不可省略 | 每个视频至少画一个框(不用精标) |

## 附录 A:图片数据集通道(run-dataset)的构建要求

如果数据以**抽帧图片**形式提供(不走 Label Studio 视频标注),输入契约是:

```
<数据集根>/
├── images/<split>/<序列名>-<帧号:06d>.jpg   # 有序帧,帧号 6 位,如 demo.mp4-000141.jpg
└── labels/<split>/<序列名>.txt              # 每行 "frame_id action_id",与 images 帧号一一对应
```

- split 目录名任意(建议 train/val/test);序列名 = 图片文件名前缀(可含 .mp4)
- 动作 id 对应 `labels/data.yaml` 的 6 类顺序(0=idle ... 5=short_brush_cleaning)
- 标签帧必须在 images/ 中有对应图片,否则 run-dataset 报错
- 参考现成样例:`datasets/cleansight-ActionMixed/images/` + `labels/`
