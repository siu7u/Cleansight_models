# 模型卡：temporal-gru

## 版本钉定

- 模型版本：v1
- 权重：`registry/gru-v1/gru-final-20260704-150629.pt`
- 数据视图：`endo-project-v1`
- YOLO 版本：`yolo-v1`
- 特征映射版本：`legacy-20d-v1`
- 特征维度：20
- 窗口长度：64

## 上线门禁

- 因果性：是
- 感受野：64 帧
- 参数量：256,131
- 单 tick 延迟：待测

## 评估结果

- Acc：68.54
- 逐类召回：
  - Idle：88.67%
  - Long_Brushing：34.73%
  - Short_Brushing：42.97%
- Edit：70.77
- F1@0.1：48.74
- F1@0.25：40.34
- F1@0.5：25.21
- 离线-在线落差：待测

## 晋升结论

- 是否晋升：否
- 阻塞原因：刷洗召回低于 70% 临时目标；单 tick 延迟和离线-在线落差尚未测量。

## 训练历史

### 20260709-132544

- 模型: `gru`
- 训练模式: `full`
- 数据集映射: `data/Endo_Project/mapping.txt`
- 特征目录: `data/Endo_Project/features`
- 标签目录: `data/Endo_Project/groundTruth`
- 训练视频索引: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`
- 测试视频索引: `[16, 17, 18, 19]`
- 输入维度: 20
- 窗口长度: 64
- 训练轮数: 10
- batch size: 32
- 学习率: 0.001
- 设备: `cuda`
- 输出权重: `weights/gru-final-20260709-132544.pt`

### 20260709-132906

- 模型: `gru`
- 训练模式: `full`
- 数据集映射: `data/Endo_Project/mapping.txt`
- 特征目录: `data/Endo_Project/features`
- 标签目录: `data/Endo_Project/groundTruth`
- 训练视频索引: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`
- 测试视频索引: `[16, 17, 18, 19]`
- 输入维度: 20
- 窗口长度: 64
- 训练轮数: 10
- batch size: 32
- 学习率: 0.001
- 设备: `cuda`
- 输出权重: `weights/gru-final-20260709-132906.pt`
