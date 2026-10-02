# 分支合并方案（草案 v1）：feat/lhh/feature-scheme × feat/roi-training

> 起草：2026-10-02 · 状态：**提案，待队友评审**（评审人：roi-training 分支负责人）
> 目的：两分支约 480 个改动文件的安全合并，核心是 **v3.1 数据登记方式的统一** 与 **两处结论矛盾的口径核查**。

---

## 1. 现状盘点

| | 我方（feature-scheme） | 对方（roi-training） |
|---|---|---|
| 改动规模 | 182 文件 | 298 文件 |
| 主线 | v3.1 数据发布 + nodep/GRU/时长先验 + 可视化 | acc+5pp 攻关 + roi-grid-v4 契约 + 架构轴排除 |
| 指标口径 | edit / F1@0.25（段级） | 帧准确率 acc / 段覆盖召回 |
| v3.1 接入 | **原地升级**既有条目（revision c30f62f5/df99637f） | **新建 -v31 条目**（revision 875d73d1），v3 条目保留（revision 6375eba） |
| 重叠文件 | 40+（testsets.yaml、actionmixed-auto manifests、data.py、models/__init__、sliding_window_pipeline、YAML_CONFIG 等） | 同左 |

**关键事实**：双方 catalog 历史已实质分叉——分叉点仍是 v1 时代条目，我方升级为 v3 原地条目（7b039550→c30f62f5），
对方改为 v3 新名条目（revision 6375eba，与我方同源数据、不同算法/时点）+ -v31 系列。**同一物理数据在三套 revision 下并存。**

## 2. 两个必须先核查的结论矛盾（合并前置）

### 2.1 v3.1 段级 edit 方向相反

| 侧 | 结论 | 条件 |
|---|---|---|
| 我方 | test edit **+1.49**（5-seed 中位 34.85 vs 33.36） | GRU + nodep-226d，v3.1 test split |
| 对方 | 段级 edit **系统性下降 4~5pp**（v4 契约 16/16 全负） | mstcn2 + roi-grid，自建 v31 split |

**核查方案（建议合并前完成）**：
1. 双方互换评测脚本与 split 定义，同一 checkpoint 上互跑（我方 benchmark.cli.eval ↔ 对方段覆盖探针）；
2. 优先排查三个假设：① 模型-特征组合敏感性（mstcn2+roi 对重标敏感、GRU+nodep 不敏感）；② edit 计算口径（对方含退化解门槛）；
③ split 差异（对方 v31 split 若含 #198/#207 旧视频则会引入不一致）；
3. 我方可提供：v3→v3.1 逐帧标签 diff（5 视频）、双版本数据本地并存。

### 2.2 对方未解释的 insert F1 −11pp 回归

- 对方观察：v3.1 上 insert F1 回归，但 train 的 insert 标签一帧未改；
- 我方线索：v3.1 变更集中 flush/sb/water（#193 sb、#194 flush、#201 water、#202 sb、#205 sb）——insert 标签确实未动，
  但 **类别权重按重标后分布重算**（compute_class_weights 每 run 自动），flush/water 帧数变化会改变 insert 的有效权重；
  另外若对方 val 含 #205（sb 细化重标），insert 的 val F1 会被边界漂移牵连；
- 建议对方按上述两条复测；我方可配合提供逐类帧数变化表。

## 3. v3.1 登记方式统一（核心决策，两个选项）

### 方案 A：版本化条目（对方方式，推荐）

- canonical 登记采用 -v31 式新条目；数据变更=新 dataset_version + 新 manifest 目录；
- 我方迁移：9 个原地升级条目（v3/roi-v1/clean-v3/hand/global-hand/cnn/hand-cnn/v4/roi-v4）改为指向 v3.1 数据的新条目
  或标注 superseded_by；我方实验引用名随之迁移（配置文件机械替换，风险低）；
- v3 数据可复现性：ModelScope 数据仓库 git 历史含 v3 完整内容（925aa7c 之前），建议**补打 tag v3 / v3.1** 固化；
- 优点：符合 registry 纪律、revision 语义清晰、对方 298 文件零迁移；缺点：我方约 15 处配置改名。

### 方案 B：原地升级（我方方式）

- 保留我方 9 条目为 canonical；删除对方 -v31 条目，其 manifest 并入主目录；
- 对方大量 run/报告引用 -v31 名，迁移成本高且其历史结论的 revision 溯源链会断；
- 优点：我方零迁移；缺点：违背【数据变更=新版本】纪律，且 git 历史保版本的可靠性依赖合并纪律。

**推荐 A**，理由：纪律一致性 > 单侧迁移成本；且 2.1 矛盾未解前，版本化条目让两套数据口径可并行复现。

## 4. 合并执行计划（三阶段）

### Phase 0：前置（1 天）
- [ ] 2.1 / 2.2 两项核查完成或至少定位到口径差异层面；
- [ ] ModelScope 数据仓库打 tag（v3 / v3.1）；
- [ ] 双方确认方案 A/B 拍板。

### Phase 1：机械合并（半天，不动语义）
- [ ] 以对方分支为 base 合入我方（对方 298 文件新代码为主干，我方以 docs/实验资产为主）；
- [ ] framework/testsets.yaml：**按方案 A 重建 catalog 段**——保留对方全部条目，我方 9 条目按 A 迁移，
      revision 以双方各自的算法重算并交叉验证（一次 validate_testsets 全绿为准）；
- [ ] benchmark/manifests/：主目录取对方版本 + 我方 v3.1 内容（train 13+1/val 3）并入 -v31 目录；
- [ ] 其余 40 重叠文件：代码类以对方为主（新增功能多），我方 sliding_window_pipeline/data.py 的改动
      （mask 分发等）逐 hunk 核对保留；docs 类双取；usage/YAML_CONFIG.md 双方条目合并去重。

### Phase 2：验证门（1 天）
- [ ] validate_testsets.py 全量：两套 revision 的条目全部 ok；
- [ ] pytest framework/tests tests -q：对齐对方基线（488 通过/7 预存失败），无新增失败；
- [ ] 关键指标复现抽检：我方 nodep 5-seed 中位数、对方 roi-grid-v4 32-seed 定标值，各抽 1-2 seed 重跑对账；
- [ ] YAML_CONFIG 与 git ls-files '*.yaml' 交叉核对。

### Phase 3：收尾
- [ ] 合并结果推 main（或团队约定的集成分支）；
- [ ] FEATURE_LAB 与对方周报互加交叉引用（含 2.1/2.2 核查结论）；
- [ ] datasets/README.md 快照节更新登记方式说明。

## 5. 风险与回退

| 风险 | 缓解 |
|---|---|
| testsets.yaml 重建引入登记错误 | Phase 2 的 validate 全绿门 + 双 revision 交叉验证 |
| 代码 hunk 合并破坏对方功能 | 对方 7 项预存失败测试为基线，不新增即过；关键路径抽检 |
| 口径矛盾未解即合并 | 允许：版本化条目保证两套口径可并行复现，不阻塞合并 |
| 我方配置改名遗漏 | 全局 grep 旧条目名 + CI/validate 兜底 |

---

*附：我方可供核查的资产——v3→v3.1 逐帧标签 diff 记录（session 归档）、双版本数据本地并存、
nodep 5-seed 全部 checkpoint 与评测 JSON、LOVO 17 折 npz（GRU/TCN 双份）。*