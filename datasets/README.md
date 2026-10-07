# 本地数据挂载

本目录只提供统一的本地挂载点，数据本体、生成的 `data.yaml` 和绝对符号链接不进入 Git。

当前 catalog 约定：

- `datasets/endo-project-v1/`：历史 Endo Project `mapping.txt / features / groundTruth / splits`。
- `datasets/cleansight-yolo/`：**现行**标准 YOLO 数据集（ModelScope `lhh010/cleansight-yolo`；
  含 `group1_large`（3 类）与 `group2_small`（5 类），各自带 train/val/test 和 `data.yaml`）。
  旧 `datasets/yolo/`（3 类旧版）已删除。
- `datasets/cleansight-ActionMixed/`：时序 ActionMixed 数据集（`labels/data.yaml` + `frames/data.yaml`；
  `framework/testsets.yaml` 的 `temporal.actionmixed-v2` 引用）。
- `datasets/cleansight-ActionMixed-auto/`：自动标注流水线时序数据集（v3，LS project-16 源；
  `temporal.actionmixed-auto-v3` 引用；ModelScope `lhh010/cleansight-ActionMixed-auto`，
  `--preset actionmixed-auto` 下载；含 `task_ids.yaml` 溯源，无图片）。
- `datasets/cleansight-ActionSequence/`：ActionSequence 数据集（按动作类分目录，各带 `data.yaml`）。
- `datasets/raw/label-studio/`：可选的 Label Studio 原始导出，不得提交。

数据身份、split 和 fingerprint 仍以 `framework/testsets.yaml` 为准；本目录只是运行时路径。

历史 Endo Project 可以用符号链接接入，不要复制进 Git：

```bash
ln -s /absolute/path/to/Endo_Project datasets/endo-project-v1
```

目标目录必须包含 `mapping.txt`、`features/`、`groundTruth/` 和 `splits/`。完成挂载后运行
`python tools/validate_testsets.py --catalog framework/testsets.yaml --json` 校验身份与清单。

## 从 ModelScope 下载

标准 YOLO 分组数据集（`lhh010/cleansight-yolo`）可用根目录脚本直接下载，产物落在
`datasets/cleansight-yolo/`（默认不入 Git）。先安装 `modelscope` 并配置 token：

```bash
pip install modelscope
# token 通过 MODELSCOPE_TOKEN 环境变量或仓库根目录 .env 提供
python download_modelscope_dataset.py --preset yolo
```

下载完成后：

- `datasets/cleansight-yolo/group1_large/`：大目标组（3 类：`hand`、`scope_control_body`、
  `scope_mid_section`）。
- `datasets/cleansight-yolo/group2_small/`：小目标组（5 类：`syringe`、`air_gun`、
  `scope_distal_end`、`short_brush`、`brush_tip_out`）。

两组各带 `data.yaml`（`path: .`，相对分组目录）。直接用 Ultralytics 时在分组目录内执行
`yolo detect train data=data.yaml ...`；经 framework 训练/评估时把实验配置的
`data.data_yaml` 指向下载的 `data.yaml` 即可。

手动 `git clone` 下载同样放这里：克隆到 `datasets/cleansight-yolo/` 后删除仓库元数据和上传
缓存（`.git/`、`.gitattributes`、各组下的 `.ms_upload_cache`），只保留数据与文档文件。注意
`git lfs pull` 需要跑完，否则 train/val 图像不完整。

## 当前划分快照（v3.1，2026-09-28 定稿 → **v3.2 fps 修正，2026-10-07**）

> 身份锁定：revision `c30f62f5`（train+val）/ `df99637f`（含 test），catalog 以 `framework/testsets.yaml` 为准；
> ModelScope `lhh010/cleansight-ActionMixed-auto`（v3.1 已发布）。逐视频 LS task id 溯源见数据集内 `task_ids.yaml`。

| split | 视频数 | 帧数 | idle | water | flush | lb_insert | lb_withdraw | sb_cleaning | 来源 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| train | 14 | 9,646 | 6,546 | 0 | 930 | 1,289 | 437 | 444 | project-16 |
| val | 3 | 3,140 | 2,080 | 0 | 184 | 495 | 206 | 175 | project-16 |
| test | 8 | 2,639 | — | — | — | — | — | — | project-18（跨批次专项） |

**train（14，LS task id → 视频前缀）**：#192 4ace5352 · #193 4cc6a009 · #194 c1367d51 · #195 5b181b9b · #196 789d58df · #197 15311df5 · #201 39da2635 · #202 071eb2d6 · **#203 f809e944（v3.1 新入集，266 帧 flush 52）** · #204 67aa31ca · #206 4894e7ba · #208 349f2a55 · #210 9c0f89a1 · #211 8634f3bc

**val（3）**：#199 1b2c95ff · #205 e8ea5bb7 · #209 52d2541c

**test（8，project-18 task#213-220）**：6f1a85e3 · 6d8c7af2 · ca09e5b5 · c7c853a4 · 935f9e44 · 67edf4f9 · ac38ca6b · 6d8b215c

### 被剔除的（隔离区，不参与训练/评估）

| 隔离区 | LS task | 视频前缀 | 帧数 | 原属 | 原因 |
|---|---:|---|---:|---|---|
| `occlusion/` | #198 | f173153a | 195 | train | 遮挡严重的 flush——特征源头缺失；留作遮挡专项改进后的回归用例 |
| `deprecated/` | #207 | 152453e5 | 244 | val | 仅含废弃动作 water_injection（该类 train/val 已全部归零，id 位置保留） |

> **v3.2（2026-10-07）**：发现 LS 时间轴 24fps 与视频实际 30fps 不一致（证据：14/37 task 标注上限精确贴 时长×24），
> 全部标签帧号按 ×1.25 重标定（帧级 43.9% 变化、段结构不变、含隔离区）；split 成员与 revision 不变（revision 仅锁成员），
> 逐类帧数 train [5798,0,1154,1603,543,548] / val [1830,0,226,612,255,217]。**v3.2 与 v3.1 及更早全部指标不可比**。
> v3.3（待发）：#221-238 共 18 个新视频（检测中）将增补入 train/val。
>
> v3.1 相对 v3 的变更：#193/194/201/202/205 标注重生成（#201 大幅重标，water 帧归零）；#198/#207 隔离；#203 补录。
> **v3.1 与 v3 指标不可比**——v3 口径的历史实验数字见 `docs/FEATURE_LAB.md`。
