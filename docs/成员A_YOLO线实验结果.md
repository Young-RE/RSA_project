# 成员A —— YOLO 线实验结果汇总

> 实验机:RTX 5060 Laptop GPU(8GB) | 日期:2026-11-03 ~ 11-04 | 本文件为报告 I.D/I.E 章节素材

## 实验设置(全组统一协议)

| 项 | 设置 |
|---|---|
| 数据 | 单帧红外, 按台风划分: train 666 / val 190(23 个台风)/ test 121 |
| 图像 | 640×640, 8-bit 灰度 |
| 训练 | ultralytics, epochs=100, batch=16, imgsz=640, seed=42, 默认增强 |
| 评估 | **统一口径**(src/eval_metrics.py): AP@IoU=0.5, AR, FROC; 中心点 ME/MAE/RMSE/MD |
| 备注 | 验证集按台风划分防泄漏; best.pt 取验证集 mAP50 最优轮 |

## 结果对比(test 集, 121 图 / 121 目标)

| 模型 | 训练方式 | AP@IoU0.5 | AR | 中心点 MAE(°) | 中心点 MD(°) |
|---|---|---|---|---|---|
| YOLOv8n(基线①) | 预训练 | 0.3298 | 0.9669 | 7.62 | 1.42 |
| YOLOv11n | 预训练 | **0.4572** | 0.9504 | 4.74 | 0.97 |
| YOLOv8n | 从零 | 0.2609 | 0.9504 | 4.78 | 0.85* |
| YOLOv8n+CBAM(改进版) | 从零 | 0.3242 | 0.8430 | 5.56 | 1.07 |
| DINO(说明文档参考) | — | 0.4438 | — | — | — |

\* 从零 YOLOv8n 仅有 27 个高置信度预测参与中心点配对, MD 参考性有限。

## 关键发现(报告讨论素材)

1. **CBAM 注意力净增益 +0.063 AP(+24% 相对提升)**:同为从零训练,0.2609 → 0.3242,
   在完全相同的训练协议下注意力模块带来稳定提升, 归因干净。
2. **预训练权重增益 +0.069 AP**:0.2609 → 0.3298, 小数据集上迁移学习的作用与
   注意力模块相当——这也是报告里可以讨论的点。
3. **YOLOv11n 超过对标基线**:0.4572 > DINO 参考值 0.4438, 且中心点 MD 0.97°。
4. **过拟合模式一致**:四个模型验证集 mAP50 均在 53~92 轮见顶后回落
   (如 yolov8n 预训练 0.4193@ep67 → 末轮 0.196), 666 张训练图下 100 epoch 偏长。
   后续实验建议 patience=20~30 或 epochs=70。
5. 从零训练收敛更慢:CBAM 最优在第 92 轮, 尚未充分收敛, 适当延长 epochs 或加
   预训练权重(加载 yolov8n backbone 后微调)预期还能涨。

## 图表位置

- 训练曲线: `runs/train/<实验名>/curves.png`(每模型一张)
- FROC 曲线: `runs/../out/eval/<实验名>_test_froc.png`
- 合并 FROC 对比: `out/eval/comparison_froc.png`
- 逐图预测(json, 供成员C 汇总): `out/eval/<实验名>_test_preds.json`

## 复现命令

```bash
python src/prepare_yolo_dataset.py
python src/train_yolo.py --model weights/yolov8n.pt --epochs 100        # 基线①
python src/train_yolo.py --model weights/yolo11n.pt --epochs 100        # 对比
python src/train_yolo.py --model src/models/cbam_yolov8.yaml --epochs 100  # 改进版(从零)
python src/train_yolo.py --model yolov8n.yaml --epochs 100 --name yolov8n_scratch_e100  # 对照(从零)
python src/evaluate_model.py --weights runs/train/<实验名>/weights/best.pt   # 统一口径评估
python src/compare_models.py                                             # 汇总对比
```
