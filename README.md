# 随机信号分析项目 —— TD 检测与定位

利用葵花8/9号红外(AHI 13通道)与 ATMS 微波(8通道)卫星数据,对热带低压(TD)进行**水平框检测 + 中心点定位**,创新方向为多帧时序动态信息。

- 任务分工:见 [小组任务分工表.md](小组任务分工表.md)
- 截止:2026-12-25(第17周)提交报告 + 代码 + 口头汇报

## 目录结构

```
随机信号分析项目/
├── dataset/                    # 数据(不进 git)
│   ├── single_frame/           # 单帧: ir/ 红外, mw/ 微波(8通道)
│   ├── Multiple_frames_6h/     # 多帧序列(还有 12h/18h), (7,640,640) float32 npy
│   ├── RSMC_Best_Track_Data/   # 台风最佳路径原始数据(可自制新标签)
│   └── 文献/                   # 11 篇给定参考文献 PDF
├── src/                        # 全部代码
│   ├── data_utils.py           # 数据读取/坐标换算(全组共用)
│   ├── visualize_labels.py     # 第4周: 图像 + bbox/中心点可视化
│   ├── dataset_stats.py        # 第4周: 数据集统计(报告素材)
│   ├── view_multiframe.py      # 第4周: 多帧序列动图
│   └── eval_metrics.py         # 统一评估指标(AP/AR/FROC/MAE/RMSE/MD, 成员C维护)
├── out/                        # 脚本输出(不进 git)
├── 小组任务分工表.md
└── README.md
```

## 环境安装

### 第4周(只需轻量库,今天就能装)

```bash
pip install numpy matplotlib pillow
```

### 第5周起(深度学习环境)

> 注意:本机默认 Python 3.14 较新,部分 ML 库可能不兼容,建议用 conda 建 Python 3.11 环境:

```bash
conda create -n td python=3.11 -y
conda activate td
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # 有 NVIDIA 显卡
# 或 CPU 版: pip install torch torchvision
pip install ultralytics opencv-python pyyaml tqdm scipy
```

各成员跑深度学习前在群里报告自己的 GPU 情况(本地显卡型号 / 服务器 / Colab)。

## 第4周启动步骤(每人必做)

1. 装环境(见上),克隆/同步本仓库(组长会建 GitHub 远程)
2. 跑通自己负责的脚本,把输出图发群里:
   - 全员:`python src/visualize_labels.py --index 0` → 输出 `out/vis/*.png`
   - 全员:`python src/dataset_stats.py` → 看懂统计表与热力图
   - 成员D 加跑:`python src/view_multiframe.py --hours 6 --index 0` → 输出 `out/vis/*.gif`
   - 成员C 加跑:`python src/eval_metrics.py`(自测通过,确认评估口径)
3. 精读分配的 3 篇文献(见分工表第五节),每篇写一段摘要发群里
4. 周日例会:交文献摘要 + 报告 GPU 情况

## 评估口径(全组合同)

- 检测:AP@IoU=0.5、AR、FROC 曲线 —— 一律调用 `src/eval_metrics.py`
- 定位:ME / MAE / RMSE / MD(像素与度数,0.11°/px)
- 对标参考:说明文档 slide16,DINO 在 IR 上 IoU=0.5 时 AP=0.4438
