# Git 分支约定（方案 C：维持现状 + 约定）

> 目的：不重写 `feat/eval-frame` 等历史分支（历史已交织、拆分风险高），但从**今天起**用
> 约定控制分支体量，避免再次出现"一个分支装下六个主题"的过载。

## 背景（为什么有这个文档）

`feat/eval-frame` 分支累计 31 个提交、269 个文件变更，混入了框架重构、时序扩展、benchmark
统一、文档体系、杂项清理等多个独立主题；`refact/train` 与之共享 17/18 提交，成为死分支。
这些历史**保持原样，不 rebase、不拆分、不删除**，只作为教训。

## 约定（新开发强制）

### 1. 一个分支只做一件事

- 分支名携带主题：`feat/<能力>`、`fix/<缺陷>`、`refactor/<范围>`、`docs/<主题>`、`chore/<杂项>`。
- 禁止在功能分支里顺带改无关文件（如顺手更新 gitignore、清理临时文件）。
- 主题之间互相独立；一个功能没做完不叠开下一个功能分支。

### 2. 分支要小且短命

| 指标 | 上限 | 超限动作 |
|---|---|---|
| 存活期 | ≤ 1 周 | 合并或暂存后删除，不长期挂机 |
| 提交数 | ≤ 10 个 | 主动 rebase/squash 归并中间提交 |
| 变更文件 | 尽量 ≤ 50 个 | 拆主题，另开分支 |

### 3. 提交纪律

- 不把 WIP 提交（如 `save: work in progress`）留在将被合并的分支上；开发中可用
  `git commit --amend` / `git rebase -i` 归并，合入前保证历史干净。
- 提交信息遵循 `<type>(<scope>): <描述>`，与仓库现有风格一致。
- 杂项（gitignore、zone.identifier、路径修复）要么 squash 进所属功能分支，要么走独立的
  `chore/` 分支，不混入功能主题。

### 4. 合并前

- 先 `git rebase master`（或 merge master）消冲突，再合入；合入后立即删除该分支。
- 删除死分支：`git branch -d <name>`（确认已合并）；本地与远程都清。

### 5. 长期分支

- 需要长期存活的分支（如发布线）应在文档中登记**用途与合并计划**，不允许"挂机无人管"。

## 2026-10-12 收口（已执行）

「臃肿」的实际形态不是分支多，而是**一条长命分支把多个主题叠了十万行**：`feat/roi-training`
领先 master **62 个提交 / 308 文件 / 102,121 插入**，且 `behind = 0`（master 是它的严格祖先）。
处置如下，**未改写任何已推送历史**。

### 1. 先把混合的工作区按主题拆开（7 个提交）

工作区里当时混着 6 个互不相关的主题（数据换代 / presence 回归修复 / 新契约 / 可视化工具 /
分析探针 / 周报），先拆成单一主题提交，再谈分支：

| 提交 | 主题 |
|---|---|
| `feat(data)`: 登记 v4.1 数据换代 | catalog + 下载器 + 配套测试 + 旗舰配置 |
| `fix(temporal)`: 修复 presence 契约加载分派被顶掉 | 1 处分支修复 + 1 条回归用例 |
| `docs(features)`: 修正 roi-grid-v5 docstring | 纯文档（128 → 192 维） |
| `feat(features)`: 新增 hand-v2 与 combo-v2/v3/v4 | 契约 + 配置 + catalog + YAML 索引 |
| `feat(tools)`: 预览视频配色对齐色带图 | 可视化工具 + skill 渲染约定 |
| `feat(tools)`: 新增四个只读分析探针 | 复评隔离 / 投票 / 手部区域筛选 / 段复核 |
| `docs(weekly)`: 补齐两周简报与模型总览 | 周报 + 状态文档 |

### 2. master 收口（fast-forward，不改写历史）

master `a6f4ec4` → `3a9e671`：两者是严格祖先关系，直接 `update-ref` 快进，
**不产生 merge 提交、不改写任何历史**。`feat/roi-training` 随即删除（内容 100% 已在 master）。

### 3. 分支清理

先给**有独有提交**的分支打 `archive/*` tag（无损）再删分支，避免"清理"变成"删代码"：

| 原分支 | 独有提交 | 处置 |
|---|---:|---|
| `feat/roi-training` | 0（已全在 master） | 删除 |
| `feat/yolo-optimize` | 0 | 删除 |
| `feat/actionmixed-v3` | 0 | 删除 |
| `feat/auto-annotate` | 0 | 删除 |
| `refact/train` | 17 | **`archive/refact-train`** tag，删分支 |
| `feat/eval-frame` | 29 | **`archive/feat-eval-frame`** tag，删分支 |
| `feat/yolo-optimize-gl` | 2 | **`archive/feat-yolo-optimize-gl`** tag，删分支 |

`lhh/develop` 保留（他人工作线，不由本仓库清理）。**远程分支未动**——删远程是对团队可见的
动作，需人工确认后另行执行；本地 `archive/*` tag 与 `origin/*` 两处都可取回。

### 4. 新工作各开一条小分支

| 分支 | 主题 | 状态 |
|---|---|---|
| `feat/yolo-brush-detect` | 链路检测侧：刷类证据为何到不了时序 | 分析已入（见 [`weeks/2026-10-10_2026-10-16/YOLO_DETECTION_SIDE_20261012.md`](weeks/2026-10-10_2026-10-16/YOLO_DETECTION_SIDE_20261012.md)）；重训进行中 |
| `feat/temporal-sbc` | 时序侧提升 | 待开 |

## 现状速查（2026-08-05，历史记录）

| 分支 | 状态 | 处置 |
|---|---|---|
| `master` | 停在 `273e2f8`（zone.identifier 清理） | 等待下一次功能合并落地 |
| `feat/eval-frame` | 31 提交领先 master，含框架重构 + YOLO 优化工具化 | 历史遗留，不重写；作为下一次合并候选 |
| `refact/train` | 18 提交，17 个已被 feat/eval-frame 包含 | 死分支；下次合并落地时删除 |
