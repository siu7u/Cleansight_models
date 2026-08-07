# 逐类最优策略分析（Per-Class Strategy）

生成时间: 2026-08-07T17:55:54

> 推荐逻辑：mAP50 主指标；差距 <0.005 时看 P/R 平衡（min(P,R)）与训练成本（yolo11n 更快）。
> mAP50 < 0.05 的类标记 **检不出**，建议淘汰转 ROI 特征融合。

## group1_large

| 类别 | 最优策略 | mAP50 | P / R | 备选 | 说明 |
|---|---|---:|---|---|---|
| hand | **yolo11s-default** | 0.860 | 0.835 / 0.875 | yolo11s-strong(0.850); yolo11n-strong(0.829) | ~75min/预设 |
| scope_control_body | **yolo11s-default** | 0.617 | 0.593 / 0.604 | yolo11n-default(0.609); yolo11s-mild(0.565) | ~75min/预设 |
| scope_mid_section | **yolo11n-default** | 0.501 | 0.752 / 0.330 | yolo11n-strong(0.491); yolo11n-mosaic_off(0.484) | ~37min/预设 |

## group2_small

| 类别 | 最优策略 | mAP50 | P / R | 备选 | 说明 |
|---|---|---:|---|---|---|
| syringe | **yolo11s-strong** | 0.483 | 0.568 / 0.457 | yolo11s-default(0.451); yolo11s-mosaic_off(0.329) | ~75min/预设 |
| air_gun | **yolo11s-mild** | 0.350 | 0.652 / 0.347 | yolo11s-mosaic_off(0.349); yolo11s-strong(0.345) | ~75min/预设 |
| scope_distal_end | **yolo11s-mosaic_off** | 0.180 | 0.426 / 0.164 | yolo11s-default(0.153); yolo11s-mild(0.137) | ~75min/预设 |
| short_brush | **yolo11s-mild** | 0.001 | 0.006 / 0.026 | yolo11s-default(0.002); yolo11s-mosaic_off(0.002) | ⚠️ **检不出**：mAP50≈0，建议淘汰该类，转 ROI 图像特征融合（roi_classification） |
| brush_tip_out | **yolo11s-default** | 0.000 | 1.000 / 0.000 | yolo11s-mild(0.000); yolo11s-mosaic_off(0.000) | ⚠️ **检不出**：mAP50≈0，建议淘汰该类，转 ROI 图像特征融合（roi_classification） |

