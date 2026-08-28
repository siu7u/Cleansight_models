# 数据集构建要求与操作指南(Label Studio 人工标注)

本指南面向**通过 Label Studio 构建/扩充训练数据集的队友**。标注产出将被
自动标注链路消费:`annotate convert`(视频链)把**人工动作标签**与 **YOLO 自动
检测框**合并成时序训练数据。**检测框可以全自动生成,动作标签必须人工标注**——
本指南的核心就是"怎么标动作标签才合格"。

参考样例(已落地):`datasets/cleansight-ActionMixed`(人工标注)、
`datasets/cleansight-ActionMixed-auto`(自动检测框 + 人工动作标签合并产物)。

## 1. 整体流程

```
① 上传视频到 Label Studio → ② 人工标注动作 timeline(不画框)→ ③ 导出 JSON
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

1. **只需要 timelinelabels(动作标签)标注动作区间** —— **不需要画框**:
   - 时序训练数据的目标框由 YOLO 自动标注(`annotate run` 逐帧检测)产出,
     人工标注只负责动作阶段;
   - convert 合并两路产物;人工导出没有检测框(无 `framesCount/duration`
     帧率锚点)时,LS 帧号按 1:1 换算(假定 LS 端按视频原始帧率标注)并打印告警;
     若标注时 LS 端帧率与视频真实帧率不同,请画任意一个框作为帧率锚点以获得精确换算
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
- 一个视频在导出里**至少有一个 timelinelabels 动作区间**(空 result 或没有
  动作标签都会被 convert 跳过,如现有 14e6fadd 等 3 个视频)

## 3. 操作步骤(Label Studio)

1. 在 Label Studio 创建/复用视频标注项目,上传视频
2. 标注模板配置 **timelinelabels**(时间线标签,选项 = 上述 6 类动作名)
3. 逐视频标注:在时间线上框选动作区间,选择动作标签;未标区间即 idle
   (**不需要画检测框**——目标框由 YOLO 自动标注)
4. 全部完成后导出 JSON

## 4. 验收清单(提交前自查)

- [ ] 每个视频都标了动作区间(有 timelinelabels)
- [ ] 动作标签名与 6 类完全一致(无拼写/大小写错误)
- [ ] 导出 JSON 中每个视频有动作标签
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

## 6. 训练数据维护注意事项

> 本节面向**在已有数据集上继续扩充/修正数据的队友**：什么动作会悄悄破坏训练数据
> 与评测可比性、以及每次数据变更必须联动更新的东西。§2 是"首次标注的硬性要求"，
> 本节是"后续维护的红线"，两者都要遵守。

### 6.1 类别覆盖与平衡(直接影响模型效果)

- **每个 split 尽量覆盖全部 6 类动作**。验证/测试集缺类会直接导致评测失真:
  当前 `cleansight-ActionMixed-auto` 的 val 缺 `air_injection`/`short_brush_cleaning`,
  test 只有 98 帧 `long_brush_withdraw` 单类——test 指标恒为 0.0,模型真实水平
  完全测不出来。新视频到位后按类别均衡重划 split 是当前最优先的维护动作。
- **少数类优先扩量**。现有 train 分布: `idle` 45%,`air_injection`/`short_brush_cleaning`
  各仅约 6%。挑选/录制新视频时优先补含 `air_injection`、`short_brush_cleaning`、
  `long_brush_withdraw` 的片段;训练配置目前没有类别权重,数据侧的平衡比什么都重要。
- **idle 不需要标**(未标注区间默认为 idle),但非 idle 动作**必须整段覆盖**:
  漏标一段 = 那段被当成 idle,直接污染标签。

### 6.2 视频与文件纪律(维护时不可破坏)

- 一个视频 = 一个连续时间序列;不要在已有视频上拼接/裁剪后换名覆盖。
- **标注后不要替换或重编码视频文件**。帧数一变,convert 的帧号换算全错位,
  标签帧数会偏少(见 §7 常见问题)。视频与标注版本必须一一对应。
- 文件名全局唯一、稳定;沿用现有命名风格 `05ba4406-clip_<起始时间戳>_<结束时间戳>.mp4`。
- 导出 JSON 中 `data.video` 文件名必须与本地视频一致(convert 只按文件名匹配)。

### 6.3 增量扩量的正确姿势

```bash
# 新视频批量自动检测(断点续跑:跳过已有产出的视频,只补新的)
python -m framework.cleansight_eval.cli.annotate run \
    --videos <新视频目录> --config framework/experiments/auto-annotate.yaml \
    --frame-stride 4 --resume

# 合并动作标签 → 训练数据(train/val/test 各跑一次,已有视频被覆盖重写、幂等)
python -m framework.cleansight_eval.cli.annotate convert \
    --annotations outputs/annotations \
    --labels-export <最新人工导出.json> \
    --out datasets/<数据集名> --split train
```

- **`--resume` 的陷阱(踩过坑)**:resume 只认"文件存在"就跳过。如果 `outputs/annotations`
  里混有之前的 smoke 产物(`--max-frames` 跑的前几十帧),该视频会被跳过、特征几乎全零。
  历史事故:05ba4406 / 4807dbbe 两个视频因此全零,已删除重跑修复。**凡是用过
  `--max-frames`/`--out` 改过参数,重跑前先确认或清空对应 JSON**。
- convert 对人工导出中缺失/无有效标注的视频**跳过并告警**(不中断);交付前确认
  结尾没有"跳过"告警,否则该视频没有进训练集。
- 图片通道(run-dataset)扩帧时,`images/` 与 `labels/` **必须同步新增**:
  标签帧在 images/ 无对应图片会直接报错。

### 6.4 版本与登记纪律(数据变更必须联动,否则评测不可信)

任何数据内容变化(增删视频、改标签、重划 split)**不只是改数据目录**,必须同步:

1. 更新 split 清单 `benchmark/manifests/<数据集>/{train,val,test}.txt`;
2. 更新 `framework/testsets.yaml` 中该数据集的 `revision`
   (三个 split manifest 拼接内容的 sha256);
3. 跑目录校验,必须全部 OK:
   ```bash
   python tools/validate_testsets.py --catalog framework/testsets.yaml --json
   ```

- **val/test 是评测锚点**:重划 split 会破坏与历史指标的对比。确需变更时保留
  val/test 不变或记录变更原因(参考 2026-08-21 的 14 视频更新:只动 train,
  val/test 保持不变以维持可比)。
- **新增/删除动作类别是全链路工程,不是加个标签名**:标签表、label mapping、
  特征维度、模型输出层、CARD.md / pin.yaml 全部要联动,通常还伴随重训。
- **自动检测只覆盖 6 类**(`short_brush`/`brush_tip_out` 恒零,40 维特征中 10 维
  永远是 0)。这是当前已知限制:检测框只当特征用,动作标签必须人工,别拿检测结果
  当动作依据。

### 6.5 交付前自查(扩充场景)

- [ ] convert 结尾无"跳过"告警
- [ ] 自动标注质量报告无"漏检/类别缺失"告警
      (`python tools/quality_report.py --auto outputs/annotations --manual <人工导出.json> --out ...`),
      并记录各检测类 presence recall/precision 供后续对比
- [ ] 统计过 train/val/test 类别分布:train 六类齐全;val/test 缺类或单类要显式说明
      (`python tools/dataset_stats.py --dataset datasets/cleansight-ActionMixed-auto \
       --manifest-dir benchmark/manifests/actionmixed-auto --json <报告路径>`),
      分布表随数据变更重新生成并留档对照
- [ ] 抽样画框预览,检测质量肉眼过关
      (`python tools/visualize_annotations.py --json ... --video ... --output ...`)
- [ ] 持久化训练数据预览核对（框位置/类别/动作标签与人工标注一致）
      (`python tools/visualize_dataset.py --dataset ... --sequence ... --images ... --output ...`),
      运行无"缺 bbox 文件"告警
- [ ] 新视频序列长度符合消费约束(GRU 的 `window: 16` 要求序列 ≥ 16 个采样帧)

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| convert 跳过某视频 | 导出中无该视频 / 无动作标签(空 result) | 回到 Label Studio 补动作标注后重新导出 |
| convert 报 KeyError(动作名) | 标签名不在 6 类内 | 检查拼写与大小写,重新导出 |
| 标签帧数偏少 | 导出后视频文件被替换(帧数变化) | 保持视频文件与标注时的版本一致 |
| convert 告警"无 LS 帧率锚点,按 1:1 换算" | 标注时未画框,且 LS 端帧率与视频真实帧率不一致 | 不影响流程(按 1:1 换算);若需精确帧号换算,标注时画任意一个框即可 |

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
