# 成员 B · DINO R50 与 RT-DETR-L 实验结果

任务是在 640 × 640 的单帧红外卫星图上检测热带低压。沿用 `prepare_yolo_dataset.py` 的按台风划分，训练集 666 张、验证集 190 张、测试集 121 张。两次训练均以 COCO 预训练权重初始化，随机种子为 42，使用 RTX 4070 Laptop GPU（8 GB）。

## 训练记录

| 模型 | 训练轮数 | 批大小 | 用于测试的权重 | 验证集 AP@0.5 | 训练记录 |
|---|---:|---:|---|---:|---|
| DINO R50 | 12 | 1 | 第 5 轮（按验证集 AP@0.5 选取） | 0.5380 | [逐轮验证指标](member_b_results/dino_r50_validation_metrics.csv) |
| RT-DETR-L | 30 | 2 | 第 15 轮（按 Ultralytics 验证指标选取） | 0.5049 | [逐轮训练与验证指标](member_b_results/rtdetr_l_training_metrics.csv) |

DINO 使用混合精度、640 × 640 输入和随机水平翻转。RT-DETR-L 的这组结果使用 640 × 640 输入，关闭 Mosaic 和混合精度。

## DINO R50 测试结果

| 测试图片 | 真值目标 | AP@0.5 | AR | 中心点 MAE | 中心点 RMSE | 中心点距离中位数 | 中心点配对数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 121 | 121 | 0.4530 | 0.9835 | 7.75° | 12.79° | 1.54° | 120 |

[DINO R50 测试集 FROC 曲线](member_b_results/dino_r50_test_froc.png)

## RT-DETR-L 测试结果

| 测试图片 | 真值目标 | AP@0.5 | AR | 中心点 MAE | 中心点 RMSE | 中心点距离中位数 | 中心点配对数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 121 | 121 | 0.3993 | 0.9174 | 6.09° | 10.90° | 1.37° | 121 |

[RT-DETR-L 测试集 FROC 曲线](member_b_results/rtdetr_l_test_froc.png)

测试指标由项目的 [统一评估脚本](../src/evaluate_model.py)计算。AR 对应脚本在 IoU 0.5 下保留全部候选预测后的召回率。中心点误差使用置信度至少为 0.25 的预测框中心计算。

## 复现步骤

实验环境为 Python 3.11、PyTorch 2.1.2 + CUDA 12.1、TorchVision 0.16.2、NumPy 1.26.4、Ultralytics 8.4.63、MMDetection 3.3.0 和带 CUDA 算子的 MMCV 2.1.0。安装 MMCV 时需匹配 PyTorch 和 CUDA 版本。将 [DINO R50 COCO 预训练权重](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth)保存为 `weights/dino_coco.pth`，将 [RT-DETR-L 预训练权重](https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt)保存为 `weights/rtdetr-l.pt`。

将课程数据放到 `dataset/single_frame/ir/`。在仓库根目录执行下列命令，将 `<配置路径>` 替换为 MMDetection 安装目录中 `.mim/configs/dino/dino-4scale_r50_8xb2-12e_coco.py` 的路径。

```bash
python src/prepare_yolo_dataset.py
python src/train_member_b_detr.py dino --weights weights/dino_coco.pth --dino-config "<配置路径>" --epochs 12 --batch 1 --name b_dino_r50_e12
python src/evaluate_model.py --predictions runs/train/b_dino_r50_e12/b_dino_r50_e12_test_preds.json
python src/train_member_b_detr.py rtdetr --weights weights/rtdetr-l.pt --epochs 30 --batch 2 --name b_rtdetr_l_e30_nomosaic
python src/evaluate_model.py --predictions runs/train/b_rtdetr_l_e30_nomosaic/b_rtdetr_l_e30_nomosaic_test_preds.json
```

训练权重和逐图预测保存在 `runs/train/<实验名>/`，评估输出保存在 `out/eval/`。
