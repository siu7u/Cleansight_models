# 本地数据挂载

本目录只提供统一的本地挂载点，数据本体、生成的 `data.yaml` 和绝对符号链接不进入 Git。

当前 catalog 约定：

- `datasets/endo-project-v1/`：历史 Endo Project `mapping.txt / features / groundTruth / splits`。
- `datasets/cleansight-yolo/`：**现行**标准 YOLO 数据集（ModelScope `lhh010/cleansight-yolo`；
  含 `group1_large`（3 类）与 `group2_small`（5 类），各自带 train/val/test 和 `data.yaml`）。
  旧 `datasets/yolo/`（3 类旧版）已删除。
- `datasets/cleansight-ActionMixed/`：时序 ActionMixed 数据集（`labels/data.yaml` + `frames/data.yaml`；
  `framework/testsets.yaml` 的 `temporal.actionmixed-v2` 引用）。
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
