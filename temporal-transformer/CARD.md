# 模型卡：temporal-transformer

## 版本钉定

- 模型版本：v1
- 权重：`registry/transformer-v1/transformer-final-20260704-161653.pt`
- 数据视图：`endo-project-v1`
- YOLO 版本：`yolo-v1`
- 特征映射版本：`legacy-20d-v1`
- 特征维度：20
- 窗口长度：64

## 上线门禁

- 因果性：是
- 感受野：64 帧
- 参数量：400,515
- 单 tick 延迟：待测

## 评估结果

- Acc：69.70
- 逐类召回：
  - Idle：94.98%
  - Long_Brushing：31.43%
  - Short_Brushing：40.71%
- Edit：66.15
- F1@0.1：46.43
- F1@0.25：41.07
- F1@0.5：33.93
- 离线-在线落差：待测

## 晋升结论

- 是否晋升：否
- 阻塞原因：刷洗召回低于 70% 临时目标；单 tick 延迟和离线-在线落差尚未测量。
