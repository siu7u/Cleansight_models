# 课题⑤⑧：换更合适的架构——五种时序架构在同一 v4 契约下的对照

> 全部批次：`framework/experiments/mstcn-actionmixed-auto-roi.yaml` 基座 + `--set` 注入
> **同一份** v4 契约（`feature_schema.version=actionmixed-roi-grid-v4`、`input_dim=128`、
> `data.dataset_ref=temporal.actionmixed-auto-roi-v4`）、同 60 epoch / lr 5e-4 / wd 1e-4 /
> `best_metric=val_edit`、**同 16 个 seed**（42,7,2026,1,2,3,4,5,11,13,17,19,23,29,31,37）。
> 读数 = `evals/*.evaluation.json` 的 `metrics.summary` 在 16 个 run 上的**中位数**（test split）。
> 配对脚本：`python tmp/acc-decomp/armcmp.py acc-roiv4,acc-roiv4-seedB <臂名>`。

## 1. 结果

| 架构（`model.type`） | 参数量 | acc | edit | f1@0.25 | f1@0.5 | 相对 mstcn2 |
|---|---:|---:|---:|---:|---:|---:|
| **mstcn2 h128**（参考，A+B 共 16 seed） | 3,310,616 | **58.24** | 43.09 | 32.31 | 14.02 | — |
| mstcn h128（单阶段 TCN，同 v4 契约） | — | 54.13 | 34.16 | 19.41 | 8.84 | **−3.94**（0/15，p=6.1e-05） |
| asformer h32（heads 4 / enc 3 / dec 2） | 86,150 | 53.69 | 27.10 | 15.47 | 7.74 | **−4.55** |
| transformer h128（d_model 128 / 4 head / 2 层） | 610,182 | 53.02 | 24.56 | 14.14 | 6.03 | **−5.22** |
| mstcn+bilstm h128（clean_mstcn_bilstm） | 1,918,610 | 51.42 | 40.53 | 28.13 | 14.89 | **−6.82** |
| fact h128（d_model 128 / 3 block） | 1,360,525 | 51.41 | 30.36 | 13.79 | 4.50 | **−6.83** |
| 平凡下界（全 idle 常数预测器） | — | 55.25 | 13.70 | 12.35 | — | −2.99 |

**结论：架构不是杠杆。** 五个架构族（多阶段精化 TCN / 注意力编码器-解码器 / 纯 Transformer /
TCN+BiLSTM 混合 / FACT 式 token 匹配）在同一份特征上相差 **4.5~6.8pp**，
且**全部不如 mstcn2**；其中 transformer 与 fact 甚至**低于 55.25 的平凡下界**。

## 2. 判读与限制

1. **单点、未调参**：每个新架构只测了**一个容量点**，没有做超参搜索（学习率、层数、
   dropout、warmup 全部沿用 mstcn2 的配方）。因此严格说这否证的是
   "**在 mstcn2 的配方下**换个架构就能更好"，不是"这些架构在任何配方下都不行"。
   考虑到差距达 4.5~6.8pp（远超 8-seed 噪声 ±0.7~1.3pp），且五种架构**无一超过参考**，
   "架构不是本轮该投的方向"这个判断是稳的。
2. **过拟合形态清晰**：transformer 的 best epoch 中位数只有 **12**（60 上限），
   asformer 为 26，而 mstcn2 多为 40~60；14 个训练视频的数据量支撑不起纯注意力架构。
   fact 的 `段数比` 1.84、mstcn+bilstm 0.72——两者都在"段结构"上跑偏。
3. **一个反直觉的好信号**：`mstcn+bilstm` 的 acc 只有 51.42（比平凡下界还低 3.8pp），
   但 **edit 40.53 / f1@0.25 28.13 / f1@0.5 14.89** 与 mstcn2 同级（43.09 / 32.31 / 14.02）。
   即：**"帧准"与"段准"可以被架构强烈解耦**——该架构几乎不猜非 idle 帧（帧准被 idle 先验吞掉），
   但一旦猜出来，段边界的质量并不差。这从架构侧独立复现了课题⑨的核心提醒：
   **acc 一个指标不足以描述模型质量**。
4. **契约增益补偿不了架构差距（已实测）**：把**同一份 v4 契约**喂给**原版单阶段 `mstcn`**，
   acc 只有 **54.13**（16 seed 配对：Δacc **−3.94pp，0/15，p=6.1e-05**；edit −9.38、
   f1@0.25 −13.17，全部 0/16）。也就是说 v4 的 +4.4pp 是在 **mstcn2 这个架构**上兑现的，
   换回更弱的架构连基线都够不上——**特征与架构不是可分离的两笔账**，
   报"v4 契约收益"时必须同时写明架构（本会话所有结论的主语都是 `mstcn2` h128/s4l10）。
   批次：`runs/acc-push/acc-roiv4-mstcn/`。

## 3. 后续动作（尚未执行）

- 若要把"架构不是杠杆"升级为强结论，需要给 transformer/asformer 各做一次
  学习率 × 层数的小网格（每点 ≥8 seed，判读门限 1.5pp），成本约 2~3 小时 GPU。

## 4. 复现命令（示例：asformer）

```bash
python tools/run_capacity_matrix.py --runs-dir runs/acc-push/acc-roiv4-asformer \
  --config mstcn-actionmixed-auto-roi.yaml --hidden 32 \
  --seeds 42,7,2026,1,2,3,4,5,11,13,17,19,23,29,31,37 \
  --best-metric val_edit --epochs 60 --no-probe-baseline \
  --set train.lr=0.0005 --set model.type=asformer --set model.heads=4 \
  --set model.num_encoders=3 --set model.num_decoders=2 \
  --set model.input_dim=128 --set feature_schema.dim=128 \
  --set feature_schema.version=actionmixed-roi-grid-v4 \
  --set data.dataset_ref=temporal.actionmixed-auto-roi-v4
```

run 目录：`runs/acc-push/acc-roiv4-{asformer,transformer,fact,mstcnbilstm}/`，
各自带 `CAPACITY_SUMMARY.md`（含逐 seed 表、噪声地板、逐类 recall）。
