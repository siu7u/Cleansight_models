# 检测侧专项：刷类证据为什么到不了时序链路（2026-10-12）

> 过程性文档，属本周（2026-10-10 ~ 10-16）。分支：`feat/yolo-brush-detect`。
> 上游结论见 [`README.md`](README.md) §2.2 与 [`COMBO_V3_SBC_20261011.md`](COMBO_V3_SBC_20261011.md) §6。

## 0. 一句话结论

**时序 `sbc`（short_brush_cleaning）恒零召回，卡在检测侧，而检测侧有两个独立问题：
① 全局 `conf: 0.25` 白扔了一部分刷类召回（test 上 **0.7669 → 0.8738**，零训练成本）；
② `brush_tip_out` 在任何阈值下召回都是 **0.0000**，且 group2 的 **val 刷类标注口径与 train
不一致（面积中位 4.12×）**，因此 val 上的刷类读数既不能用来判模型、也不能用来选点。**

本轮**没有**得到"更好的 g2 权重"——先量清了该不该训、按什么选点，避免把 GPU 时间花在
被污染的选择口径上。

## 1. 链路位置与动机

时序特征来自检测框（`datasets/cleansight-ActionMixed-auto-lhh/frames/<split>/*.txt`），
由 `cli.annotate` + `framework/experiments/auto-annotate-yolo11.yaml` 用**全局** `conf: 0.25`
筛出。而 `sbc` 是时序侧召回恒 0 的类，其 oracle 天花板此前量过 = **+2.37 acc / +12.91 edit**
（[`COMBO_V3_SBC_20261011.md`](COMBO_V3_SBC_20261011.md) §6），是全仓库最大的一块可见空间。

## 2. 事实一：sbc 帧里的刷类框覆盖率（只读标签，可复现）

在 v4.1 auto 数据上逐帧对齐「动作标签 = sbc」与「该帧是否含刷类框（short_brush / brush_tip_out）」：

| split | sbc 帧数 | 其中带刷类框 | 覆盖率 |
|---|---:|---:|---:|
| train | 1,374 | 237 | **17.2%** |
| val | 974 | 213 | **21.9%** |
| **test** | 127 | **0** | **0.0%** |

train 全集刷类框只有 short_brush 550 + brush_tip_out 368 个；test 全集只有 24 + 32 个。
**82.8% 的 train sbc 帧、100% 的 test sbc 帧在时序输入里没有任何"刷子"证据**——
这与此前"sbc 与 idle 在手部区域里静态构成几乎逐项相同"的结论一致：不是特征没取对，
是证据没进来。

## 3. 事实二：置信度阈值是真实杠杆，但对两个刷类效果相反

`tools/probe_detection_conf_recall.py`（新增，一次前向 + 离线套阈值）在 group2_small 上的读数
（`yolo11s-g2-default-best.pt`，imgsz 640，贪心匹配 IoU ≥ 0.5）：

| conf | syringe | air_gun | scope_distal_end | **short_brush** | **brush_tip_out** |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 0.5238 | 0.3208 | 0.7635 | **0.9195** | **0.0000** |
| 0.05 | 0.4842 | 0.3038 | 0.7404 | **0.8738** | **0.0000** |
| 0.15 | 0.4186 | 0.2901 | 0.7296 | 0.8282 | 0.0000 |
| **0.25**（现行） | 0.3665 | 0.2560 | 0.7249 | **0.7669** | 0.0000 |
| 0.50 | 0.2817 | 0.1741 | 0.7040 | 0.5431 | 0.0000 |

（split = test；GT 实例数 syringe 884 / air_gun 293 / scope_distal_end 2774 / short_brush 1403 / brush_tip_out 21）

- `short_brush`：**0.25 → 0.05 召回 +10.7pp**（0.7669 → 0.8738），代价是其他类各掉 ~2~4pp。
  即"逐类阈值"（`auto-annotate*.yaml` 里已预留的 `conf:` 字典形态）对刷类是**正收益**，
  可以零训练成本先吃到。
- `brush_tip_out`：**任何阈值都是 0.0000**。这是一个**权重不认这个类**的问题，降阈值无用
  （只会放噪声）。

## 4. 事实三：group2 的 val 刷类标注口径与 train 不一致（本轮最关键的负面发现）

先用最近框 IoU 区分"没出框"与"出框但对不上"（conf=0.001 取全部预测）：

| split | 类 | GT | 到最近**同类**框 IoU ≥0.5 | 到最近**任意类**框 IoU ≥0.3 |
|---|---|---:|---:|---:|
| val | short_brush | 461 | 10.8% | 37.7% |
| **test** | short_brush | 1403 | **94.3%** | 95.8% |
| val | brush_tip_out | 456 | **0.0%** | 66.2% |
| test | brush_tip_out | 21 | 0.0% | 4.8% |

`short_brush` 在 val 上召回 0.03、在 test 上 0.92 —— **同一权重、同一类，差 30 倍**。
于是查标注本身（`tools/probe_detection_label_geometry.py` 新增，只读标签）：

| 类 | train 面积中位 | val（相对 train） | test（相对 train） |
|---|---:|---:|---:|
| syringe | 0.01219 | 0.01321 (**1.08×**) | 0.00870 (0.71×) |
| air_gun | 0.02314 | 0.02535 (**1.10×**) | 0.01999 (0.86×) |
| scope_distal_end | 0.00328 | 0.00359 (**1.09×**) | 0.00343 (1.05×) |
| **short_brush** | 0.00351 | 0.01443 (**4.12×**) | 0.00419 (1.20×) |
| **brush_tip_out** | 0.00322 | 0.00624 (**1.94×**) | 0.00659 (**2.05×**) |

前三个类的三个 split 口径都稳定在 0.71~1.10×；**只有两个刷类偏离 1.9~4.1×**。
`short_brush` 在 val 上的框比 train 大 4 倍，模型学的是 train 的尺度 → 在 test（1.20×）上
命中 94%，在 val（4.12×）上塌到 10.8% ≥0.5。

**推论（对流程的影响）**：
1. **group2 的 val 不能用来判刷类能力，也不能用来早停/选点**——它会系统性地把"模型其实会"
   读成"模型不会"。此前 `external_checkpoints/EXPERIMENTS` 里 g2 的 4 个预设 val mAP50 全在
   0.11~0.19（[`../../../external_checkpoints/EXPERIMENTS/LOG.md`](../../../external_checkpoints/EXPERIMENTS/LOG.md)），
   有很大一部分可能是这个口径问题，而不是增强策略的差异；**这批结论需要重判**。
2. `brush_tip_out` 在 val 有 456 个 GT、**0.0%** 被判成自己这一类，但 66.2% 在附近**有框**
   （任意类 ≥0.3）→ 它在被邻近类吸收。加上 train 只有 572 框、test 只有 21 框，
   这是个**不可测也不可训**的类。给时序链路用的现实选择是把它并入 `short_brush`
   （或按"刷子"单一类下发）；但**改类别表就改了特征契约的类序/维度**，按
   `modelset-quality` 的规则属于新特征映射版本、时序侧需重训 —— 本轮**不做**，只登记为待决。

## 5. 环境约束（已固化为配方）

本机是 **12GB RAM 的 WSL**（swap 已近满）。`small_s_copy_paste` 的 960px + `workers: 8` +
`batch: 12` 在本机**连续两次在 epoch 0 之前被内存杀掉**（run 目录只剩 `args.yaml`、GPU 0%）。
新增低内存配方 `small_s_copy_paste_lowmem`（`workers: 2` / `batch: 8`，其余与
`small_s_copy_paste` 逐项一致），1 epoch 预检**跑通**（含 val，显存 4789 MiB，GPU 88%）。
`run_experiment` 相应支持 `workers` 透传（原先被丢弃）。

## 6. 本地能做与不能做的边界（重要）

- **能做**：在 `datasets/cleansight-yolo/group2_small` 上训练/评测检测器（本地有 14,916 /
  4,746 / 3,592 张图）。
- **不能做**：**本地重跑时序数据的自动标注**。`datasets/cleansight-ActionMixed-auto-lhh`
  是 ModelScope 数据集的 git clone，**只有 `frames/`（检测框 txt）+ `labels/`（动作标签）+
  `task_ids.yaml`，不含图片也不含视频**；43 个 auto 视频在 `datasets/cleansight-ActionMixed/images`
  里**零交集**（该目录的 16 个视频是旧的手动数据集）。全盘搜索也只有 txt。

  → **"换成更好的 YOLO 权重重新生成 frames/"这一步必须在持有视频的机器上执行**。本机只能
  给出"换成什么权重、用什么阈值、预期提升多少"的配方与度量。

## 7. 复现命令

```bash
# 事实一的覆盖率表（只读标签，秒级）
# 事实三的口径检查（只读标签，秒级）
python tools/probe_detection_label_geometry.py --group group2_small --json tmp/geom_g2.json

# 事实二的阈值曲线（一次前向 + 离线套阈值；约 1~2 分钟）
python tools/probe_detection_conf_recall.py \
    external_checkpoints/EXPERIMENTS/best_weights/yolo11s-g2-default-best.pt \
    --group group2_small --split test val --out tmp/conf_recall_g2.json

# 低内存配方训练（12GB WSL 可用；本机在跑）
PYTHONPATH=framework python -m framework.cleansight_eval.cli.sweep \
    --group group2_small --preset small_s_copy_paste_lowmem --device 0
```

## 8. 下一步（交接）

1. **先修口径再训模型**：`short_brush` 的 val 标注需人工复核（4.12× 面积偏离）；在修好之前，
   group2 的一切刷类选点改用 train 内部分折，或改用与 train 同口径的 test。
2. **逐类阈值先落地**：`auto-annotate*.yaml` 把刷类阈值降到 0.05（其他类保持 0.25），
   **在持有视频的机器上**重生成 frames/ 后，重测 sbc 帧的刷类覆盖率（当前 train 17.2%）。
3. `brush_tip_out` 的归属（并入 `short_brush` vs 补标注 vs 退登记）待决；改动即新特征映射版本。
4. 在跑的 `small_s_copy_paste_lowmem`（200 轮）用于回答"补刷类数据/ copy_paste 能否把
   train 口径下的刷类召回再抬一档"——**选点不得用 val 的刷类读数**。
