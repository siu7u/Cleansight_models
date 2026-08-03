# 本地数据挂载

本目录只提供统一的本地挂载点，数据本体、生成的 `data.yaml` 和绝对符号链接不进入 Git。

当前 catalog 约定：

- `datasets/endo-project-v1/`：历史 Endo Project `mapping.txt / features / groundTruth / splits`。
- `datasets/yolo/group1_large/`：大目标 YOLO 数据集。
- `datasets/yolo/group2_small/`：小目标 YOLO 数据集。
- `datasets/raw/label-studio/`：可选的 Label Studio 原始导出，不得提交。

数据身份、split 和 fingerprint 仍以 `benchmark/testsets.yaml` 为准；本目录只是运行时路径。

历史 Endo Project 可以用符号链接接入，不要复制进 Git：

```bash
ln -s /absolute/path/to/Endo_Project datasets/endo-project-v1
```

目标目录必须包含 `mapping.txt`、`features/`、`groundTruth/` 和 `splits/`。完成挂载后运行
`python tools/validate_testsets.py --catalog benchmark/testsets.yaml --json` 校验身份与清单。
