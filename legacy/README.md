# Legacy 历史快照

本目录保存统一 framework 建立前的独立训练、推理和数据流水线，仅用于审计历史实现与复现旧
实验。它不是新开发入口，也不得被 `framework/`、`benchmark/` 或实验 YAML 作为运行依赖。

边界：

- 新训练和模型推理只通过 `framework`。
- 正式评测、feed-mode、指标和报告只通过 `benchmark`。
- 仍需交付的模型资产位于顶层 `registry/`。
- 本地数据挂载位于顶层 `datasets/`。
- legacy 中的脚本、绝对路径、旧指标和旧 split 不获得兼容性保证。

历史目录：

- `temporal-gru/`
- `temporal-causal-tcn/`
- `temporal-transformer/`
- `yolo-detection/`

迁移映射：

| 迁移前 | 迁移后 | 当前状态 |
|---|---|---|
| `temporal-*/` 的独立模型、训练与推理脚本 | `legacy/temporal-*/` | 冻结；历史网络兼容实现已进入 `framework/.../temporal/models/` |
| `temporal-*/registry/`、根目录 CARD/pin/report | `registry/temporal/<model-version>/` | 活跃模型资产真源 |
| `yolo-detection/registry/` | `registry/detection/` | 活跃检测资产真源 |
| `yolo-detection/pipeline/datasets/` | `datasets/yolo/` | 本地数据挂载，Git 忽略 |
| `yolo-detection/pipeline/` 其余脚本 | `legacy/yolo-detection/pipeline/` | 冻结，仅供审计 |
| `tools/eval_temporal_detailed.py` 等直接模型工具 | `legacy/tools/` | 冻结；统一评测改由 `benchmark` 调用 `framework` |
| 根目录 `utils/temporal_main.py` | `legacy/utils/` | 冻结；活跃代码不得 import |

如需复现旧命令，应从保留该布局的 Git 历史提交或发布 tag 建立独立环境，不要把新功能继续
写入本目录。
