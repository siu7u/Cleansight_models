# 坏段排查记录：表现差的视频段 → Label Studio 只读核查（2026-10-09）

> **触发**：周会要能"指着具体视频段说问题"。本文把 v4.1 全量重训里**表现最差的视频段**挑出来，
> 逐条拿到 Label Studio 上核对原标注，记录可能的问题。
>
> **一句话结论**：坏段**不是标注时间轴的问题**——LS 原始标注与本地标签的 ×1.25 换算关系
> 在 **8/8 个 test 视频、逐段**上成立（并用视频本体帧率独立验证）；坏段的共同特征是
> **输入侧证据缺失**：判别类检测框（syringe / air_gun / short_brush / brush_tip_out）
> 在 test 动作段里几乎不出现。另在 **val 视频 #205** 上查到 LS 标注侧的**重叠/嵌套重复段**。
>
> **本文所有 Label Studio 访问均为只读**（`/api/tasks/*` GET + 视频文件下载），未写入、未修改任何标注。

## 0. 口径与限制（先读这一节）

| 项 | 口径 |
|---|---|
| 模型 | `runs/v41/flagship-v41`（v4.1 旗舰：mstcn2 s4l10 h128 × `roi-v4-v41` 128 维），**best.pt**（`val_edit` 选点） |
| seed | 该臂 8 条 best.pt 评测中 **acc 最接近中位**的一条 = `mstcn2-20261008-170149`，acc **64.38**（8 seed 中位 64.91） |
| 数据 | `temporal.actionmixed-auto-roi-v4-v41.test`，8 视频 / 5,272 帧 / revision `4aa9bf31…` |
| 时间轴 | 本地 `frame_id` 是 1-based 真实视频帧号（30 fps 视频每 2 帧采 1 帧）→ 秒 = `frame_id / 30` |
| 限制 1 | 坏段清单来自**单 seed**（中位 seed）。逐视频差异极大（见 §3.4），**不能把单 seed 的段级结论当成"8 seed 一致"** |
| 限制 2 | §4 的"输入侧证据缺失"是**相关性证据**，不是因果证明；因果要在特征层做干预实验 |
| 限制 3 | 跨臂对照只用各臂自己的中位 seed（与仓库可视化约定一致），未做 seed 配对检验 |

## 1. 排查方法（可复现）

1. **挑坏段**：从 best.pt 预测产物算逐视频 acc + 逐类帧召回 + 逐 GT 段的命中率
   （段内同类命中帧 / 段长），列出 `coverage = 0`（整段漏）与"预测该类但 GT 无该类"（误报）。
2. **上 LS 核对**：只读 API 取原标注（`timelinelabels` 的 `ranges`），与本地标签**逐段**对齐；
   同时扫重叠段、零长段、<0.5s 碎段，统计面覆盖 project 16/18 全部 task。
3. **看视频**：从 LS 取回原始 mp4（`/data/upload/<project>/<file>`），`ffprobe` 核帧率/帧数/时长，
   并在失败段抽关键帧目视；进一步可用 `tools/visualize_predictions.py` 把**逐帧预测动作叠加到视频上**
   （顶部横幅 = 当前预测动作、下方小字 = GT，不符标 `MISMATCH`），产出见 §8.3。
4. **输入侧画像**：对每个 GT 动作段统计逐检测类框数/帧（`frames/<split>/<video>-<frame:06d>.txt`）。

复现（**已入库工具** `tools/ls_segment_review.py`，只读、不加载 checkpoint）：

```bash
python tools/ls_segment_review.py compare --split test        # LS 原标注 ↔ 本地标签 + 标注质量扫描
python tools/ls_segment_review.py failures --arm flagship-v41  # 逐视频 acc / 逐段命中率
```

本次排查中另有一批临时脚本在 `tmp/ls_check/`（`arm_video_matrix.py` 臂 × 视频矩阵、
`det_evidence.py` 段内检测证据、`lsclient.py` 只读客户端），未入库；关键产物 CSV 见 §8.2。
色带图用官方工具生成：`tools/plot_gt_vs_pred_bands.py`（含逐视频 acc）。

## 2. 结论 1：×1.25 口径经 LS 原标注 + 视频本体双重证实（**这不是坏段的成因**）

v4.1 说明数据集的标签时间轴做了 **×1.25** 修正，成因自述是"LS 时间轴 24 fps vs 视频 30 fps"。
本次直接从 LS 取原标注核对：

| 视频 | LS 段数 | 本地动作段数 | 段数一一对应 | 逐段起点位移比（本地秒 ÷ LS帧/24） |
|---|---:|---:|---|---|
| `67edf4f9`（flush×6） | 6 | 6 | ✅ | 1.0133 / 1.0052 / 1.0034 / 1.0015 / 1.0020 / 1.0016 |
| `6d8b215c`（flush×1） | 1 | 1 | ✅ | 1.0000 |
| `6d8c7af2`（lbi×2, lbw×2） | 4 | 4 | ✅ | 1.0008 / 1.0005 / 1.0035 / 1.0008 |
| `6f1a85e3`（lbi×1, sbc×5） | 6 | 6 | ✅ | 1.0051 / 1.0033 / 1.0102 / 1.0000 / 1.0000 / 1.0012 |
| `935f9e44`（lbi×3, lbw×3） | 6 | 6 | ✅ | 1.0028 / 1.0004 / 1.0008 / 1.0010 / 1.0004 / 1.0000 |
| `ac38ca6b`（lbi×3, lbw×3） | 6 | 6 | ✅ | 1.0016 / 1.0000 / 1.0007 / 1.0026 / 1.0010 / 1.0006 |
| `c7c853a4`（sbc×3） | 3 | 3 | ✅ | 1.0171 / 1.0118 / 1.0031 |
| `ca09e5b5`（lbi×2, lbw×1） | 3 | 3 | ✅ | 1.0006 / 1.0005（首段 2.4000 见下） |

- 比值 ≈ 1.000（= 本地 `frame/30` ÷ `LS帧/24`）⇔ **本地帧号 = LS 帧号 × 1.25**，逐段成立。
- **首段 2.4000 是取样量化，不是缺陷**：LS 首段从帧 1 起（0.042 s），而本地采样起点是第 3 帧（0.10 s），
  比值 = 0.1/0.0417 ≈ 2.4。8 个视频里 3 条 2.4 全部出现在"LS 首段起点 = 帧 1"的段上，已逐条确认。
- **视频本体独立验证**（LS 取回的原始 mp4，`ffprobe`）：

| 视频 | 帧率 | 帧数 | 时长 | 本地标签最大帧号 |
|---|---:|---:|---:|---:|
| `6d8b215c` | 30/1 | 468 | 15.60 s | 467 |
| `6f1a85e3` | 30/1 | 1290 | 43.00 s | 1289 |
| `ac38ca6b` | 30/1 | 2137 | 71.33 s | 2136 |

  即本地标签的帧号**逐帧对齐 30 fps 视频本体**，而 LS 导出帧号只有它的 0.8 倍 → **LS 导出的帧号是 24 fps 时间轴的帧号**。
- **顺带确认一个易踩的坑**：在 LS 网页上看，时间显示是对的（LS 用"帧号 ÷ 24"换算显示秒）；
  出错的环节只有一个——**导出/合并时把 LS 帧号直接当成视频帧号**（旧 v3.x 口径即如此，导致动作段整体偏早 20%）。

> 结论：8 个 test 视频的**标注时间轴没有问题**，坏段不能归因到"标签错位"。

## 3. 结论 2：坏段清单（test，旗舰中位 seed）

### 3.1 逐视频读数（同一模型、同一口径）

| 视频 | 时长 | acc | flush | lb_insert | lb_withdraw | sbc |
|---|---:|---:|---:|---:|---:|---:|
| `6d8b215c` | 15.6 s | **0.376** | **0.000** | — | — | — |
| `ac38ca6b` | 71.3 s | **0.400** | — | 0.226 | 0.349 | — |
| `67edf4f9` | 30.6 s | 0.577 | 0.485 | — | — | — |
| `935f9e44` | 85.9 s | 0.714 | — | 0.667 | 0.569 | — |
| `6f1a85e3` | 43.0 s | 0.735 | — | 0.516 | — | **0.000** |
| `c7c853a4` | 8.5 s | 0.766 | — | — | — | 0.444 |
| `ca09e5b5` | 37.4 s | 0.766 | — | 0.583 | 0.740 | — |
| `6d8c7af2` | 59.1 s | 0.779 | — | 0.557 | 0.457 | — |

pooled acc 64.38 背后：**最好的视频 0.779、最差的 0.376**（差 40pp）。

### 3.2 整段未检出（GT 段内同类命中帧 = 0）——共 12 段（其中**动作段 10 段**、idle 段被整段误判 2 段）

| 视频 | 类别 | 时间码（秒） | 段长 | GT 段内模型实际输出 |
|---|---|---|---:|---|
| `6d8b215c` | flush | **4.17–13.83** | 146 帧 | idle（**该段占视频 62%**） |
| `ac38ca6b` | lb_insert | 50.03–56.63 | 100 帧 | idle + lb_withdraw |
| `935f9e44` | lb_withdraw | 74.17–79.37 | 79 帧 | idle |
| `6d8c7af2` | lb_insert | 55.23–59.03 | 58 帧 | idle + lb_withdraw |
| `ac38ca6b` | lb_withdraw | 43.17–46.97 | 58 帧 | idle + lb_insert |
| `6f1a85e3` | sbc | 5.77–7.37 | 25 帧 | idle |
| `6f1a85e3` | sbc | 8.50–9.90 | 22 帧 | idle |
| `6f1a85e3` | sbc | 1.63–2.50 | 14 帧 | idle |
| `6f1a85e3` | sbc | 7.50–8.30 | 13 帧 | idle |
| `6f1a85e3` | sbc | 5.10–5.57 | 8 帧 | idle |
| `c7c853a4` | （idle 误报） | 5.17–5.37 | 4 帧 | 预测 sbc |
| `6d8c7af2` | （idle 误报） | 59.10 | 1 帧 | 预测 lb_withdraw |

补充：**误报段 18 个**（预测某动作类、GT 段内无该类），其中最集中的是 `ac38ca6b`（9 个，例如把
lb_insert 段整段预测成 lb_withdraw：31.17–36.23；把 idle/lb_withdraw 段预测成 lb_insert：40.23–45.63、61.17–64.10）。
另有 **15 个"部分命中"段**（coverage 0.06~0.49），例如 `67edf4f9` 的 flush 4.43–8.10 s 只命中 23%、
`ac38ca6b` 的 lb_insert 5.30–18.30 s 只命中 20%。

### 3.3 三个失败模式（把 12 段归类）

| 模式 | 表现 | 涉及段 | 备注 |
|---|---|---|---|
| **M1 稀类短段全漏** | 0.5–1.5 s 的 `sbc` 段整段漏 | `6f1a85e3` **5/5 段 coverage 全 0**；`c7c853a4` 3 段 coverage 0.059 / 0.385 / 0.933（未全漏但两段几乎全丢） | sbc 在 test 共 8 段 / 127 帧，是"少而碎"的一类 |
| **M2 insert↔withdraw 方向互换** | 相邻同类段标反，成对出现（漏一个 + 在同位置误报另一个） | `ac38ca6b`（2 段整段漏 + 9 段误报，其中 5 段与方向互换有关）、`6d8c7af2`（1 段整段漏 + 2 段误报均为方向互换） | 与 v3 时代"方向信息不在特征里"的现象一致，v4.1 上仍未解决 |
| **M3 flush 整段漏** | 9.67 s 的 flush 全判 idle | `6d8b215c` | 该视频 62% 是 flush，模型却给出"全 idle"解 |

### 3.4 这些坏视频不是某个臂的毛病（22 臂 × 8 视频矩阵）

同一视频、不同臂的 acc 跨度（各臂取自己的中位 seed，best.pt）：

| 视频 | 22 臂中位 | 最差臂 | 最好臂 | 跨度 |
|---|---:|---|---|---:|
| `6d8b215c` | 0.395 | arch-transformer 0.000 | **feat-boxset 0.816** | 0.816 |
| `67edf4f9` | 0.483 | arch-transformer 0.000 | feat-roi-v5 0.760 | 0.760 |
| `ac38ca6b` | 0.454 | arch-transformer 0.203 | feat-global-hand 0.658 | 0.455 |
| `c7c853a4` | 0.727 | arch-transformer 0.258 | feat-combo 0.930 | 0.672 |
| `ca09e5b5` | 0.731 | arch-transformer 0.333 | feat-global-hand 0.948 | 0.615 |
| `6d8c7af2` | 0.791 | arch-transformer 0.158 | feat-global-hand 0.903 | 0.745 |

**`6d8b215c` 最能说明问题**（GT：idle 0.03–4.10 s / **flush 4.17–13.83 s** / idle 13.90–15.57 s）：

| 臂 | 该视频 acc | 预测段 |
|---|---:|---|
| 旗舰 `flagship-v41` | **0.376** | `idle 0.03–15.57`（**全 idle**） |
| `feat-roi-v5` | 0.402 | idle 0.03–6.83 / flush 6.90–7.23 / idle 7.30–9.10 / **sbc 9.17–10.77** / idle 10.83–15.57 |
| `feat-global-hand` | 0.397 | idle 0.03–4.43 / flush 4.50–5.10 / idle 5.17–9.23 / **sbc 9.30–10.97** / idle 11.03–12.97 / lbi 13.03–14.17 / idle 14.23–15.57 |
| **`feat-hand`** | **0.795** | flush 0.03–2.30 / idle 2.37–4.03 / **flush 4.10–14.63**（几乎完全命中）/ idle 14.70–15.57 |
| **`feat-boxset`** | **0.816** | idle 0.03–3.97 / **flush 4.03–8.50** / lbi 8.57–8.83 / **flush 8.90–10.63** / idle 10.70–11.90 / **flush 11.97–14.97** / idle 15.03–15.57 |

→ 同一个段，hand-only 与 boxset 契约都能（分段地）检出来，而 `roi-v4` 这一支给的是"全 idle"：
**证据在输入里，但 `roi-v4` 这套通道没有把它送到模型面前**。

## 4. 结论 3：坏段的共同特征是"输入侧证据缺失"

对 test 全部 **35 个 GT 动作段**统计逐检测类框数/帧（`frames/` 检测产物，v4.1 与 v3 逐坐标相同）：

| 动作（段数） | 框/帧 | hand | scope_control | scope_mid | scope_distal | syringe | air_gun | short_brush | brush_tip_out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flush（7） | 4.09 | 2.00 | 1.00 | 1.00 | 0.00 | **0.05** | **0.04** | 0.00 | 0.00 |
| lb_insert（11） | 3.99 | 1.99 | 1.00 | 0.97 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| lb_withdraw（9） | 3.89 | 2.00 | 1.00 | 0.89 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sbc（8） | 2.81 | 1.97 | 0.76 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 |

三条可复述的观察：

1. **判别类检测基本不触发**：`syringe` / `air_gun` / `short_brush` / `brush_tip_out` 在 test 动作段里
   逐段中位都是 **0.00~0.05 框/帧**——包括本该由"短毛刷"支撑的 **8 段 sbc 全部 0.00**。
   即：flush 与 sbc 这两类在**输入特征里没有直接证据**，只能靠 hand/scope 框的**几何与数量变化**去推。
2. **同一视频内，动作段与 idle 段的检测组成可以完全相同**：`67edf4f9` 逐 0.5 s 轨迹显示
   idle（0–4.4 s）与 flush（4.4–8.1 s）都是 `hand 2 / scope_control 1 / scope_mid 1`；
   `syringe` 只在 3.0 s、5.5 s 各闪现 1 帧。该视频 flush 召回 0.485 —— 模型学的显然不是"看到注射器"。
3. **漏检段普遍落在"框更少"的时段**（相关性，单 seed）：
   `6d8b215c` 的 flush 段 3.33 框/帧（其他 6 段 flush 为 3.99~4.22）；
   `935f9e44` 74.17–79.37 的 lbw 段 3.49（其他 lbw 3.97~4.19）；`ac38ca6b` 50.03–56.63 的 lbi 段 3.62。
   `6d8b215c` 段内唯一可见的变化是 `hand 1→2`、`scope_mid 1→0`（发生在 3.4 s 前后，早于标签起点 4.17 s）。

## 5. 结论 4：Label Studio 标注侧确实有问题——但在 train/val，不在 test

扫过 project 18（`action-test`）9 个 task 与 project 16（`action-train`）44 个 task 的全部标注：

| 项目 | 已标注 task | 重叠段对 | 零长段 | <0.5 s 碎段 |
|---|---:|---:|---:|---:|
| **18 = action-test**（test 8 视频） | 8 | **0** | **0** | **0** |
| **16 = action-train**（train/val） | 37 | **7** | **1** | **1** |

- **test 侧标注是干净的**（task 212 未标注，且不在 `task_ids.yaml` 里，属正常）。
- **val 视频 #205（`e8ea5bb7`，项目 16）有 6 对重叠**：
  `sbc 68–399`（2.83–16.62 s）**内嵌** 4 条 sbc（`71–97`/`134–171`/`232–272`/`294–320`），
  flush 两对交叠（`774–803` 与 `780–806`、`891–919` 与 `898–919`）。
  本地导出对重叠取**并集**：LS 8 段 flush → 本地 6 段、LS 6 段 sbc → 本地 2 段。
  **该视频在 val split 里，而选点口径是 `val_edit`** → 重复段会改变 val 读数，进而影响选点。
- train #197（`15311df5`）有 1 对：`sbc 505–600` 与 **`sbc 599–599`（零长段）**。
- **sbc 被切得很碎**（LS 原始标注就这样）：`6f1a85e3` 5 段（0.54 / 0.88 / 1.67 / 0.83 / 1.46 s，
  段间只隔 2、3、4 个 LS 帧 = 0.08 / 0.13 / 0.17 s）；`c7c853a4` 3 段（间隔 4、7 帧）。
  在 LS 上按 1.25 换算成视频时间后，这几段与“一段连续短毛刷清洁”几乎连成一片 ——
  **模型面对的是 0.5 s 级碎片，而段级指标把每一片各自记账**（这也是 M1 的一部分成因）。
- 顺带记录两条**平台侧观察**（与本次数据无关，仅备案）：
  项目 16 里 task 200 / 440 / 441 / 442 显示"无标注"；
  平台上另有 project 19 `handpos-train`（2026-10-06 建，200 task）——上游似已在建新的标注工程。

## 6. 可能的问题清单（按归属，供会上讨论）

**A 输入/特征侧（最可能的主因）**
1. 判别类检测框几乎不触发（§4 表），`flush` / `sbc` 在特征层没有直接证据 → 稀类段必然难学；
2. ROI 契约 `roi-v4` 只保留 hand/scope 的统计量，而"能检出 `6d8b215c` flush"的臂是 hand-only 与 boxset ——
   提示**证据在框的时序几何里**，`roi-v4` 的通道选择丢了它；
3. 框数偏少的时段（单手/遮挡）更容易整段漏，属可疑的分布外情形。

**B 标注侧**
4. val #205 的重叠/嵌套重复段（已确认 6 对）+ train #197 零长段 → 导出取并集，段边界被改写；
5. sbc 被切成 0.5–1.7 s 碎片（间隔 0.08–0.3 s），会让"段召回"这类验收指标失真；
6. train/val 的标注者不同（#3/#4/#5/#6/#1 五人），段粒度习惯不一致（如 #205 的 13.8 s 长 sbc）。

**C 口径/验收侧**
7. pooled acc 掩盖坏视频：同一臂逐视频 0.376~0.779，同一视频跨臂 0.000~0.816 → 交付前必须逐视频看；
8. 段级指标对碎片敏感；建议验收口径加"**稀类段覆盖率**（sbc/flush）+ 逐视频 acc"两条；
9. `val_edit` 选点已被证明会在部分臂上反噬（arch-transformer 差 −17.12pp），val 又只有 7 个视频且含 #205 缺陷。

**D 流程侧**
10. LS 帧号 = 24 fps 时间轴，导出必须 ×1.25；已核对 8/8 test 视频，但**换视频/改帧率后必须重做这一步**；
11. LS 允许重叠 timeline 段且无最小时长约束 → 建议导出前加 QA（重叠、零长、<0.5 s）或约定最小段长；
12. 上游 README 与 `task_ids.yaml` 口径不一致（v3.1 revision / 14-3-8 split），登记时只能信 `task_ids.yaml`。

## 7. 建议的下一步（会上拍板项）

1. **`flush` / `sbc` 去留**：若下游需要，按 §4 结论优先做"检测侧补类"（syringe/air_gun/short_brush 召回）
   或换契约（hand-only / boxset 已在单视频上显示能检出）；
2. **val #205 处置**：请上游修正重叠段后重导出，或本地先按"并集"显式登记该 val 视频为缺陷样本；
3. **验收口径**：把"逐视频 acc + 稀类段覆盖率"加进交付判据；
4. **LS 侧 QA**：给导出流程加重叠/零长/碎段检查（本次脚本可直接复用）。

## 8. 附录

### 8.1 LS 深链（只读，点击即到该视频）

| split | 视频 | task | 链接 |
|---|---|---|---|
| test | `6d8b215c`（M3 flush 全漏） | 220 | http://49.234.120.241:8080/projects/18/data?task=220 |
| test | `ac38ca6b`（M2 方向互换） | 219 | http://49.234.120.241:8080/projects/18/data?task=219 |
| test | `6f1a85e3`（M1 sbc 5/5 全漏） | 213 | http://49.234.120.241:8080/projects/18/data?task=213 |
| test | `c7c853a4`（sbc 碎段） | 216 | http://49.234.120.241:8080/projects/18/data?task=216 |
| test | `935f9e44`（lbw 漏段） | 217 | http://49.234.120.241:8080/projects/18/data?task=217 |
| test | `6d8c7af2`（方向互换） | 214 | http://49.234.120.241:8080/projects/18/data?task=214 |
| test | `67edf4f9`（flush 对照） | 218 | http://49.234.120.241:8080/projects/18/data?task=218 |
| test | `ca09e5b5` | 215 | http://49.234.120.241:8080/projects/18/data?task=215 |
| val | `e8ea5bb7`（**重叠重复标注**） | 205 | http://49.234.120.241:8080/projects/16/data?task=205 |

### 8.2 产物与复现命令

```bash
# ① 入库工具：LS 原标注 ↔ 本地标签 逐段对拍 + 标注质量扫描（重叠 / 零长 / <0.5s 碎段）
python tools/ls_segment_review.py compare --split test
#    → outputs/ls_review/ls_vs_local_test.csv、outputs/ls_review/ls_annotation_qa_test.csv
# ② 入库工具：逐视频 acc / 逐段命中率（best.pt，取 acc 最接近中位的 seed）
python tools/ls_segment_review.py failures --arm flagship-v41
#    → outputs/ls_review/seg_failures_flagship-v41.csv
# ③ 本次临时脚本（tmp/ 会被清理，数值已固化进本文）
python tmp/ls_check/arm_video_matrix.py   # → tmp/ls_check/arm_video_matrix.csv（22 臂 × 8 视频）
python tmp/ls_check/det_evidence.py       # → tmp/ls_check/seg_detection_evidence.csv（段内检测证据）
# ④ 色带图（GT vs 4 臂，含逐视频 acc）
python tools/plot_gt_vs_pred_bands.py --runs runs/v41 \
  --arm flagship-v41 --arm feat-roi-v5 --arm feat-hand --arm feat-boxset --axis seconds
# ⑤ 取视频核帧率（LS 上的原始 mp4：<host>/data/upload/<project>/<file>）
ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate,nb_frames,duration -of default=noprint_wrappers=1 <mp4>
```

> Label Studio 凭证来自仓库根 `.env`（`LS_HOST` / `LS_TOKEN`）。`LS_TOKEN` 是 refresh token，
> 需先 POST `/api/token/refresh/` 换 access token（约 5 分钟有效，`tools/ls_segment_review.py` 已内置）；
> 本次所有查询均为只读。

### 8.3 叠加预测视频（"视频进程里显示当前预测动作"）

| 产物 | 内容 |
|---|---|
| `outputs/visualizations/pred_v41/<序列>_pred.mp4` | 旗舰臂（中位 seed）在 8 个 test 视频上的叠加视频：顶部横幅 = 当前**预测**动作（动作色填充）、下方小字条 = **GT**（不符时标 `MISMATCH`）、画面内叠加检测框；输出 15 fps = 实时播放 |
| `outputs/visualizations/pred_v41_cmp/feat-hand_6d8b215c_pred.mp4`、`feat-roi-v5_6d8b215c_pred.mp4` | 同一视频（`6d8b215c`，M3 flush 全漏）在 3 个契约下的对照 |
| `outputs/visualizations/pred_v41/preview_6d8b215c_3arms.png` | 九宫格预览：行 = 臂（旗舰 0.376 / roi-v5 0.402 / hand 0.795），列 = t=2 / 7 / 12 s |
| `outputs/visualizations/pred_v41/_src/` | 从 LS 取回的 8 个源视频（本数据集不含像素，视频只在 LS 上） |

```bash
# 逐帧预测动作叠加到原视频（只读预测产物，不加载 checkpoint）
python tools/visualize_predictions.py \
  --artifact runs/v41/flagship-v41/h128/mstcn2-20261008-170149/artifacts/full_sequence_temporal-mstcn2-20261008-170942.predictions.json \
  --dataset datasets/cleansight-ActionMixed-auto-lhh \
  --video outputs/visualizations/pred_v41/_src/<序列>.mp4 --sequence <序列>.mp4 \
  --split test --fps 15 --out-dir outputs/visualizations/pred_v41
```

> 横幅配色与 `outputs/visualizations/bands/gt_vs_pred_tc_p01.png`（色带图）**同色**：
> idle 蓝 / flush 橙 / long_brush_insert 绿 / long_brush_withdraw 红 / short_brush_cleaning 紫。
