# yolo-test 标注分工与 BENCHMARK 等价类覆盖

最新状态快照：2026-07-31 20:23（Asia/Shanghai）
文档修订时间：2026-07-31 20:42（Asia/Shanghai）

当前等价类明细见
[equivalence_classes_20260731.md](equivalence_classes_20260731.md)。

## 1. 当前结论

- Label Studio 项目：`yolo-test`（project id `15`）。
- 现有任务：62 个。
- 项目类别：`hand`、`short_brush`、`syringe`、`air_gun`、
  `scope_control_body`、`scope_mid_section`、`scope_distal_end`、`brush_tip_out`。
- 当前已有 61 个 annotation，覆盖 61/62 个任务（98.4%）。
- 尚未标注：task 140。
- 61 个已标任务均含目标矩形轨迹，共 316 条轨迹。
- 其中 58 个任务同时具备合法机位和光照 tag，可归入 15 个精确等价类；
  另有 3 个任务需要补齐 tag 后再锁定。
- cam 状态需分开理解：task 99 是已标任务中唯一漏选 viewpoint 的任务；task 140
  尚无 annotation，因此未进入 cam 统计；`cam3=0` 表示当前没有 cam3 样本，不是某个任务漏标。

`framework/testsets.yaml` 目前没有“等价类”字段，只登记数据集、类别、split 和用途。
本文中的组合数量来自本机数据快照推导，不替代 BENCHMARK 真源。

## 2. 现场分工

完整任务表见 [task_assignment.csv](task_assignment.csv)。

| 批次 | 主标任务 | 数量 | 复核 |
|---|---|---:|---|
| A | 92–113（不含 98） | 21 | B |
| B | 114–134 | 21 | C |
| C | 135–140、146–159 | 20 | A |

执行规则：

1. 主标人员完成视频目标轨迹，并在任务表填写 `equivalence_class`。
2. 复核人员检查类别、轨迹起止、遮挡后重现、框漂移和漏标；不得直接覆盖主标结论，
   有分歧时在 `notes` 留痕。
3. 正样本必须至少有一个有效矩形轨迹；负样本必须使用项目已有选择项明确保存，
   不能用“没有 annotation”代替负样本。
4. 所有 8 类均按项目 ontology 标注；不能因为当前 BENCHMARK 仅评 6 类而漏掉
   `short_brush` 或 `brush_tip_out`。
5. task 92 先由 A、B 双人确认两个 choices 的业务含义，再决定是否计为完成。

完成门槛：

- 62/62 个任务均有明确的正/负标注结论；
- 62/62 个任务均完成交叉复核；
- 正样本的轨迹类别属于 8 类 ontology，关键帧顺序合法，框坐标不越界；
- 负样本不是空任务；
- 导出后重新统计类别、组合、空任务和复核状态，再生成锁定版本。

## 3. BENCHMARK 当前实际覆盖

统计口径：

- “图片/有框/空标/缺标”按本机 `datasets/cleansight-yolo/**` 当前文件统计；
- “框数/含类图片数”中，前者是目标框数，后者是至少出现该类的图片数；
- 当前数据目录被 Git 忽略，数字只代表现场快照。

| 数据集 | split | 图片/有框/空标/缺标 | 类别：框数/含类图片数 | 推定实际源视频 |
|---|---|---|---|---:|
| group1 | train | 5503/5503/0/0 | hand 10146/5388；control 4965/4965；mid 4653/4653 | 13 |
| group1 | val | 1595/1595/0/0 | hand 3028/1592；control 1281/1281；mid 1400/1400 | 7 |
| group1 | test | 489/489/0/0 | hand 978/489；control 157/157；mid 342/342 | 2 |
| group2 | train | 3813/3813/0/0 | syringe 1424/1424；air_gun 440/440；distal 2898/2898 | 10 |
| group2 | val | 451/451/0/0 | syringe 346/346；air_gun 33/33；distal 343/343 | 4 |
| group2 | test | 194/194/0/0 | syringe 53/53；air_gun 0/0；distal 194/194 | 1 |

所有 split 中：畸形标注行 0、越界类别 0、非法归一化框 0、孤儿 label 0。

### 3.1 test 的对象出现组合

group1 test 共 4 个组合：

| 等价类（单帧对象出现组合） | 图片数 |
|---|---:|
| hand + scope_mid_section | 265 |
| hand + scope_control_body | 80 |
| hand + scope_control_body + scope_mid_section | 77 |
| hand | 67 |

group2 test 共 2 个组合：

| 等价类（单帧对象出现组合） | 图片数 |
|---|---:|
| scope_distal_end | 141 |
| syringe + scope_distal_end | 53 |

当前 test 合计只覆盖 6 个“组内单帧对象组合”。group2 test 完全没有 `air_gun`，
且只有 1 个推定实际源视频，不能视为等价类覆盖完整。

### 3.2 BENCHMARK 与 yolo-test ontology 对照

| yolo-test 类别 | BENCHMARK 数据集 | 当前 test 含类图片数 | 状态 |
|---|---|---:|---|
| hand | group1 | 489 | 已覆盖 |
| scope_control_body | group1 | 157 | 已覆盖但源视频少 |
| scope_mid_section | group1 | 342 | 已覆盖 |
| syringe | group2 | 53 | 已覆盖但数量少 |
| air_gun | group2 | 0 | 缺失 |
| scope_distal_end | group2 | 194 | 已覆盖 |
| short_brush | 无 | 0 | BENCHMARK 未登记 |
| brush_tip_out | 无 | 0 | BENCHMARK 未登记 |

## 4. 已采集历史等价类

仓库当前可见的最新 Label Studio 历史导出是
`references/label_studio/project-10-at-2026-07-07-19-32.json`，共 17 个任务。
7 月 4 日的 13 个任务已包含在其中，不能重复累加。

### 4.1 标注完整性

| 等价类 | 任务数 |
|---|---:|
| 动作 + 检测 | 11 |
| 仅检测 | 3 |
| 空结果 | 3 |

### 4.2 动作覆盖

| 动作 | 覆盖任务数 | 片段数 |
|---|---:|---:|
| long_brush_insert | 8 | 18 |
| long_brush_withdraw | 6 | 16 |
| flush | 2 | 22 |
| short_brush_cleaning | 3 | 5 |
| air_injection | 2 | 3 |

### 4.3 任务级检测对象组合

缩写：`H=hand`、`C=scope_control_body`、`M=scope_mid_section`、
`D=scope_distal_end`、`BT=brush_tip_out`、`Sy=syringe`、
`A=air_gun`、`SB=short_brush`。

| 等价类 | 任务数 | 历史 task id |
|---|---:|---|
| 空 | 3 | 62、63、64 |
| H+C+D+M | 3 | 53、55、60 |
| BT+H+C+D+M+Sy | 2 | 61、68 |
| BT+H+C+M | 1 | 50 |
| H+C+M | 1 | 51 |
| BT+H+C+D+M | 1 | 52 |
| A+BT+H+C+D+M | 1 | 54 |
| A+H+C+D+M+Sy | 1 | 56 |
| H+C+SB | 1 | 58 |
| A+H+C+M | 1 | 59 |
| H | 1 | 69 |
| H+C+D+M+SB+Sy | 1 | 75 |

以上是 12 个任务级检测组合。它表示“整段视频中出现过的类别集合”，不能证明这些对象在同一帧、
同一动作片段内同时出现。正式等价类需在 yolo-test 标注完成后，按轨迹与动作区间相交重新统计。

## 5. 数据质量阻断项

现有校验器按文件名推导帧键，官方校验结果为 `ok`，但内容哈希发现跨 split 重复：

| 数据集 | train/val | train/test | val/test |
|---|---:|---:|---:|
| group1 | 360 | 58 | 98 |
| group2 | 161 | 58 | 0 |

部分重复图片的标签坐标还不一致：group1 train/val 164 张、train/test 58 张；
group2 train/val 94 张、train/test 17 张。split 内也有重复副本：group1 train 608 张、
val 100 张；group2 train 476 张。

因此在锁定新测试集前必须：

1. 用图片内容 SHA-256 去重，而不是只按文件名去重；
2. 同一画面存在多份标注时先仲裁坐标，不能任意保留一份；
3. 重新按源视频或明确的场景分组划分 train/val/test；
4. 将图片与 label 内容纳入可复现 fingerprint；
5. 补齐 `air_gun`、`short_brush`、`brush_tip_out` 的 test 覆盖，再讨论覆盖完成。

## 6. 复核命令

```bash
PYTHONDONTWRITEBYTECODE=1 python tools/validate_testsets.py \
  --catalog framework/testsets.yaml --json
```

```bash
git ls-files datasets/cleansight-yolo/group1_large datasets/cleansight-yolo/group2_small
git check-ignore -v datasets/cleansight-yolo/group1_large/data.yaml
```

正式导出 yolo-test 后，应追加一轮统计：任务完成率、正/负样本数、8 类任务/轨迹/关键帧数、
对象组合数、源视频数、跨 split 内容哈希重叠和标注分歧。
