# yolo-test 已采集等价类汇总

快照时间：2026-07-31 20:23（Asia/Shanghai）
文档修订时间：2026-07-31 20:42（Asia/Shanghai）

数据来源：Label Studio `yolo-test`，project id `15`，实时 JSON 导出。
本文件统计的是标注数据覆盖，不是模型 benchmark 结果。

## 1. 统计口径

当前项目把整条已裁测试片作为一个场景段：

```text
精确等价类 = viewpoint + 该任务全部 ec_tags
```

例如：

```text
cam2 + normal_light + fast_blur
```

`normal_light`、`dark`、`overexposed` 作为光照基准，要求每个任务恰好选择一个；
`fast_blur`、`defocus`、`glare_water`、`cluttered_bg` 等作为可叠加条件。

忽略 `viewpoint` 后得到“场景条件组合”；保留 `viewpoint` 后得到最终精确等价类。

## 2. 标注完成度

| 项目 | 数量 |
|---|---:|
| Label Studio 总任务 | 62 |
| 已有 annotation | 61 |
| 未标任务 | 1：140 |
| 已标任务中的目标轨迹 | 316 |
| 已标任务中的 choices 结果 | 119 |
| 具备合法机位和光照 tag 的任务 | 58 |
| 需要补 tag 的已标任务 | 3 |
| 有效场景条件组合（忽略机位） | 11 |
| 有效精确等价类（包含机位） | 15 |

61 个已标任务全部具有目标矩形轨迹，没有“只有 tag、没有目标轨迹”的空 annotation。

## 3. 已采集精确等价类

以下 15 类同时具有唯一机位和唯一光照基准，共覆盖 58 个任务。

| 精确等价类 | 任务数 | task id |
|---|---:|---|
| cam2 + normal_light | 15 | 113、119、127、128、130、131、133、134、135、147、148、149、150、152、153 |
| cam1 + normal_light | 14 | 94、122、123、124、125、136、137、138、139、146、151、156、157、158 |
| cam2 + normal_light + fast_blur | 10 | 114、115、116、117、118、120、121、129、132、159 |
| cam2 + normal_light + cluttered_bg | 4 | 104、109、110、111 |
| cam1 + normal_light + fast_blur | 3 | 100、103、126 |
| cam1 + normal_light + fast_blur + defocus + cluttered_bg | 2 | 93、97 |
| cam1 + normal_light + cluttered_bg | 2 | 101、154 |
| cam1 + normal_light + fast_blur + defocus + glare_water + cluttered_bg | 1 | 92 |
| cam1 + normal_light + fast_blur + glare_water + cluttered_bg | 1 | 95 |
| cam1 + normal_light + fast_blur + defocus | 1 | 96 |
| cam2 + dark + cluttered_bg | 1 | 105 |
| cam2 + dark | 1 | 106 |
| cam2 + dark + fast_blur | 1 | 107 |
| cam2 + dark + fast_blur + cluttered_bg | 1 | 108 |
| cam2 + normal_light + fast_blur + defocus | 1 | 112 |

前三个等价类合计 39 个任务，占 58 个有效任务的 67.2%，当前分布明显偏向
`cam1/cam2 + normal_light` 和 `cam2 + normal_light + fast_blur`。

## 4. 忽略机位后的场景条件组合

| 场景条件组合 | 任务数 |
|---|---:|
| normal_light | 29 |
| normal_light + fast_blur | 13 |
| normal_light + cluttered_bg | 6 |
| normal_light + fast_blur + defocus + cluttered_bg | 2 |
| normal_light + fast_blur + defocus | 2 |
| normal_light + fast_blur + glare_water + cluttered_bg | 1 |
| normal_light + fast_blur + defocus + glare_water + cluttered_bg | 1 |
| dark | 1 |
| dark + fast_blur | 1 |
| dark + cluttered_bg | 1 |
| dark + fast_blur + cluttered_bg | 1 |

## 5. 单 tag 覆盖

单 tag 数量允许重叠，不能直接相加。

### 5.1 机位

| tag | 已标任务数 |
|---|---:|
| cam1 | 26 |
| cam2 | 34 |
| cam3 | 0 |
| 未选择机位 | 1 |

这里的“未选择机位”只指已存在 annotation、但漏选 `viewpoint` 的 task 99。
task 140 尚无 annotation，因此没有进入上述 cam 统计；它完成标注时仍需选择一个机位。
`cam3=0` 表示当前没有任何任务被标记为 cam3，是采集/覆盖缺口，不是漏选机位的任务数。

### 5.2 场景条件

| tag | 已标任务数 |
|---|---:|
| normal_light | 54 |
| dark | 4 |
| overexposed | 0 |
| fast_blur | 22 |
| defocus | 5 |
| glare_water | 2 |
| cluttered_bg | 13 |
| similar_distractor | 0 |
| diff_scope_model | 0 |
| diff_operator | 0 |

## 6. 目标类别覆盖

“任务数”表示至少包含该类别一条轨迹的任务数；“轨迹数”表示该类别实例轨迹总数。

| 目标类别 | 任务数 | 轨迹数 |
|---|---:|---:|
| hand | 58 | 119 |
| short_brush | 4 | 4 |
| syringe | 13 | 13 |
| air_gun | 29 | 30 |
| scope_control_body | 59 | 60 |
| scope_mid_section | 56 | 56 |
| scope_distal_end | 33 | 33 |
| brush_tip_out | 1 | 1 |

61 个已标任务共形成 18 种“任务级目标出现组合”。该组合只表示整段视频内出现过哪些类别，
不等同于目标在同一帧共现。

## 7. 待补 tag 的任务

| task id | 当前已有 tag | 问题 |
|---|---|---|
| 99 | 无 | 已有目标轨迹，但缺 viewpoint、光照和场景 tag；这是当前唯一已标但没有 cam 的任务 |
| 102 | cam1、fast_blur、cluttered_bg | 缺光照基准 |
| 155 | cam1 | 缺光照和场景 tag |

task 139、156、157 已补为 `cam1 + normal_light`。加上未标的 task 140，当前还有
4 个任务未达到“可锁定等价类”状态。

## 8. 当前覆盖缺口

- `cam3`：0 个任务。
- `overexposed`：0 个任务。
- `similar_distractor`：0 个任务。
- `diff_scope_model`：0 个任务。
- `diff_operator`：0 个任务。
- `dark`：只有 4 个任务，且全部来自 `cam2`。
- `defocus`：5 个任务，全部同时带有 `fast_blur`，没有独立失焦样本。
- `glare_water`：2 个任务，全部与 `cam1 + normal_light + fast_blur + cluttered_bg`
  组合出现，没有独立水渍/反光样本。
- `brush_tip_out`：只有 1 个任务、1 条轨迹。
- `short_brush`：只有 4 个任务、4 条轨迹。

## 9. 锁定前质量检查

1. 完成 task 140。
2. 补齐 task 99、102、155 的 tag。
3. 确认是否确实没有 `cam3`、`overexposed`、`similar_distractor`、
   `diff_scope_model`、`diff_operator`；如果采集过但未打 tag，应补标。
4. 对 `brush_tip_out`、`short_brush` 做专项复核，确认低数量不是漏标。
5. 当前项目配置中的目标控制名为 `object_labels`，已有导出的 316 条轨迹结果使用
   `from_name=objects`。锁定或重导入前应确认该名称差异不会导致标注结果无法回显。
6. 补齐后重新导出，并以导出文件哈希和日期生成正式数据集版本。
