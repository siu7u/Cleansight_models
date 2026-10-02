# 实验报告：帧准确率攻关第 6 轮 —— **通道语义替换**：v4 契约，全栈读数 59.15（+5.23pp）（2026-09-29）

> **目标**：`acc` 53.92 → 58.92（+5pp）。
> **本轮成果**：新增特征契约 `roi-grid-v4`（**通道语义替换**），把帧准确率推到
> **全栈 59.15（比基线 +5.23pp）**、**单 seed 标准评测 57.59（+3.67pp，8/0，p=0.0078）**。
> 本轮是**第一次在 val 上也大幅领先**的改进（val 64.33 / 66.73 vs 基线 57.09 / 58.66）。

---

## 1. 起点与动机

第 5 轮在**源域**（与 test 无关）确认了两条事实：

1. 现行 v2 契约的 `presence` 通道**完全冗余**：`presence ≡ count>0`，
   `2ch(count+max_area)` 与 `3ch` 的逐折 Δ **完全相同**；
2. `max_area` 不可省（去掉 −2.63pp）。

→ 每个区域的**有效信息只有 2 个数**，第三、四个通道槽位应放**新信息**，而不是继续放冗余量。
第 5 轮只试过"已有 3 通道的子集"，**从没试过替换通道**——本轮补上。

## 2. 尝试：通道语义源域筛选（与 test 无关，先选型再上 test）

口径：auto **train** 14 视频、10 折视频级留出、HistGB；布局固定为 v2（高频 3×3、低频全局 1 区），
**只换通道组合**，故是单变量对照。参考 = `p,c,a`（现行 v2）。

| 通道组合 | 维度 | Δ 中位 | 配对 vs 参考 | p |
|---|---:|---:|---|---:|
| **`count, max_area, w, h`（本轮的 v4）** | **128** | **−2.53** | **+0.82pp，8/2 胜** | **0.084** |
| `count, w, h` | 96 | −2.48 | +0.90pp，9/1 | 0.027 |
| `count, max_area, w, h, dx, dy` | 192 | −2.27 | +1.09pp，9/1 | 0.037 |
| `count, max_area, mean_area` | 96 | −2.87 | +0.53pp，7/3 | 0.131 |
| `count, max_area`（去冗余 presence） | 64 | −3.40 | ±0.00（逐折相同） | — |
| `p,c,a`（现行 v2，参考） | 96 | −3.40 | — | — |
| `count, max_area, sum_area, mean_area` | 128 | −3.72 | −0.05pp | 0.820 |

**选型纪律**：本轮的产物（v4 = `count, max_area, w, h`）是**只看上表**（源域 + 维度不过大）定下来的，
**没有看 test**——这是对第 5 轮教训（"只在 test 上试出来的配置不可交付"）的直接执行。

**为什么是它**：`max_area = w×h` 把框压成一个标量，**丢掉了长宽比**；
而器械的形状是有判别力的（细长的内镜 vs 方正的刷子）。v4 用 `w`、`h` 两个通道把形状补回来。

## 3. 结果

### 3.1 与基线、与 v2 的对照（8 seed，仓库标准评测协议：`best.pt` + 逐帧 argmax）

| 契约 | 布局 | 通道 | 维度 | micro acc 中位 | 摆幅 |
|---|---|---|---:|---:|---:|
| `roi-grid-v1`（基线） | 8 类统一 2×3 | p,c,a | 144 | 53.92 | 4.32 |
| `roi-grid-v2` | 高频 3×3 | p,c,a | 96 | 56.48 | 3.45 |
| **`roi-grid-v4`（本轮）** | **高频 3×3** | **c,a,w,h** | **128** | **57.59** | **2.77** |
| `roi-grid-v3` | 高频 4×4 | p,c,a | 159 | 49.30 | 9.13 |

配对检验（8 seed，Wilcoxon）：

| 对比 | Δ 中位 | 胜/负 | p |
|---|---:|---|---:|
| **v4 vs v1（基线）** | **+3.67pp** | **8/0** | **0.0078** |
| **v4 vs v2** | **+1.06pp** | **6/1** | **0.0781** |

**v4 的 seed 间摆幅同时最小（2.77，v2 3.45、v1 4.32）**——不只是均值更高，也更稳。

### 3.2 全栈读数（集成 + val 选点解码）

| 配置 | val | test | Δ vs 基线 53.92 |
|---|---:|---:|---:|
| v4 单 seed 中位 | — | 57.59 | +3.67 |
| v4 概率集成（8 seed，零超参） | 64.33 | 58.39 | +4.47 |
| **v4 集成 + val 选点最小时长 d=15** | **66.73** | **59.15** | **+5.23** |
| 目标 | — | 58.92 | +5.00 |

`d=15` 是**在 val 上选的**（val 曲线 64.33 → 66.73，+2.40pp）；`veto` 未被 val 选中（不启用）。
**所有组件都在 val 上有支持**，不是只在 test 上试出来的。

### 3.3 val 侧对照（证明改进不是 test 特有）

| 契约 | val micro（集成，d=1） | val micro（集成，d=15） |
|---|---:|---:|
| v1 基线 | 57.09 | 58.66 |
| **v4** | **64.33** | **66.73** |

**v4 在 val 上领先基线 +7.24 / +8.07pp**，与 test 上的 +4.13 / +4.89pp 同向且更大 →
**这是 5 轮以来第一次"val 与 test 同向且都大幅改善"的改动**（第 5 轮那批"test 涨、val 跌"的候选已全部作废）。

## 4. **+5pp 的口径拆解（必须与读数一起引用）**

| 对比口径 | Δ | 是否达标 |
|---|---:|---|
| **绝对指标**：基线 53.92 → v4 全栈 59.15 | **+5.23pp** | ✅ |
| 同一推理协议（集成 + val 选点 d=15）：v1 54.26 → v4 59.15 | +4.89pp | ❌ |
| v1 的**最优**栈（veto+mindur15，第 1 轮）55.40 → v4 59.15 | +3.75pp | ❌ |
| **单 seed 同协议（仓库标准评测）53.92 → 57.59** | **+3.67pp** | ❌ |
| 8 seed 集成：54.11 → 58.39 | +4.28pp | ❌ |

**结论**：**指标绝对值已越过 58.92**，但"严格同协议"的改进量是 **+3.67 ~ +4.89pp**，
其中 +1.56pp 来自**推理侧**（8 模型集成 + val 选点平滑）——那套做法在第 1 轮就已具备，
只是当时没有并到基线读数里。
→ **按最严格口径（模型/训练本身、单 seed、仓库标准评测）仍差 1.33pp。**

**另一条必须记录的保留意见**：全栈的**逐视频**配对检验（n=8 视频，v4 栈 vs v1 栈）
Δ 中位 **+0.00pp、胜/负 3/2、p=0.3125**（3 个视频打平，2 个大胜，2 个大负）。
即 micro（按帧加权）的 +4.89pp 并非"每个视频都变好"，而是集中在少数长视频上。
逐 **seed** 的配对检验（单 seed 口径）则是 **8/0、p=0.0078**，显著。
两者不矛盾（配对维度和统计量不同），但读 micro 时必须知道它按帧加权。

## 5. 下一步（第 7 轮候选，已用源域选好）

源域筛选中还有两个**显著**优于 v2 的候选（§2 表），可作为下一轮的单变量外推：

| 候选 | 维度 | 源域 Δ vs 参考 | p | 说明 |
|---|---:|---:|---:|---|
| `count, max_area, w, h, **dx, dy**` | 192 | **+1.09pp，9/1** | **0.037** | v4 + **格内亚格位置**（把网格丢掉的亚格位置补回来，且不增加格子数，绕开"4×4 每格过小"的问题） |
| `count, w, h` | 96 | +0.90pp，9/1 | 0.027 | 连 `max_area` 也去掉（w×h 已含面积，但保留形状） |

推荐先做 **v4 + `dx,dy`（192 维）**：它是源域收益最大且显著的候选，
机制上也最"新"（亚格位置是当前所有契约都没编码过的信息）。

## 6. 产物与复现

```bash
source ../CleanSightBackend/.venv/bin/activate
python tmp/acc-decomp/channelscreen.py       # §2：源域通道筛选（与 test 无关，先选型）
python tools/run_capacity_matrix.py --runs-dir runs/acc-push/acc-roiv4 \
  --config mstcn-actionmixed-auto-roi.yaml --hidden 128 --seeds 42,7,2026,1,2,3,4,5 \
  --best-metric val_edit --epochs 60 --no-probe-baseline --set train.lr=0.0005 \
  --set model.type=mstcn2 --set model.num_stages=4 --set model.num_layers=10 \
  --set model.input_dim=128 --set feature_schema.dim=128 \
  --set feature_schema.version=actionmixed-roi-grid-v4 \
  --set data.dataset_ref=temporal.actionmixed-auto-roi-v4
python tmp/acc-decomp/final_sweep.py --runs-dir runs/acc-push/acc-roiv4   # §3.2 解码（val 选点）
```

**新增产物**：契约 `framework/cleansight_eval/temporal/features/roi_bbox_v4.py`；
登记 `temporal.actionmixed-auto-roi-v4`（`framework/testsets.yaml` + `usage/YAML_CONFIG.md` +
`docs/features/README.md`）；单测 `framework/tests/test_roi_features_v4.py`（7 条）；
训练臂 `runs/acc-push/acc-roiv4/`（8 seed）；脚本 `tmp/acc-decomp/channelscreen.py`。

**验证**：`tools/validate_testsets.py` 全量 `ok: true / errors: []`；
全量回归 **454 passed / 7 failed**（7 项均为预先存在，与改动前逐项一致）。

**已知限制**：test 仅 8 视频 / 2639 帧、val 仅 4 视频；§4 的逐视频保留意见必须随读数引用；
`v3`/`global-hand` 等已证伪契约的结论不受本轮影响。
