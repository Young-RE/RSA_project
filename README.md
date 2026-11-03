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

> 注意:`pip install torch` 默认装 **CPU 版**;有 NVIDIA 显卡必须装 cu 版。
> RTX 50 系(Blackwell)需要 **cu128 及以上**。

**实测可用安装步骤(Windows + Python 3.14, 本机 RTX 5060 验证通过):**

```bash
# 1. 预装依赖(setuptools<82 是 torch 的硬性要求; 清华源会 403 它, 用阿里云源)
pip install "setuptools<82" -i https://mirrors.aliyun.com/pypi/simple/
pip install sympy jinja2 filelock fsspec networkx typing-extensions -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. curl 下载 CUDA 版 torch 轮子(SJTU 镜像 ~9MB/s; pip 直装曾遇到空文件, 用 curl 更稳)
mkdir -p %TEMP%\torchwheels && cd %TEMP%\torchwheels
curl -sSL --retry 5 -C - -o torch-2.11.0+cu128-cp314-cp314-win_amd64.whl "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp314-cp314-win_amd64.whl"
curl -sSL --retry 5 -C - -o torchvision-0.26.0+cu128-cp314-cp314-win_amd64.whl "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/torchvision-0.26.0%2Bcu128-cp314-cp314-win_amd64.whl"

# 3. 本地安装(依赖已预装, --no-deps 防 pip 再去源里拉)
pip install --no-deps %TEMP%\torchwheels\torch-2.11.0+cu128-cp314-cp314-win_amd64.whl %TEMP%\torchwheels\torchvision-0.26.0+cu128-cp314-cp314-win_amd64.whl

# 4. ultralytics
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 验证(首次 import 约 1-2 分钟, 属正常)
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> 其他 Python 版本:把命令里的 `cp314` 换成对应版本号(如 3.11 → cp311),torch 版本号可在
> https://mirror.sjtu.edu.cn/pytorch-wheels/torch/ 目录里查。

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

## YOLO 训练框架(成员A)

三条命令跑通基线① YOLOv8n:

```bash
python src/prepare_yolo_dataset.py                              # 1. 准备数据(每台机器一次)
python src/train_yolo.py --model weights/yolov8n.pt --epochs 100  # 2. 训练(需 GPU/CPU + torch 环境)
python src/evaluate_model.py --weights runs/train/yolov8n_e100/weights/best.pt   # 3. 统一口径评估
```

- 预训练权重下载(GitHub 直连不通时用镜像):
  `curl -L -o weights/yolov8n.pt "https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/yolov8n.pt"`
- 数据集按**台风**划分 train/val(默认 20% 台风进 val),防止同一台风连续帧泄漏
- 对比模型换参数即可:YOLOv11n 用 `--model yolo11n.pt`;改进版传自定义模型 yaml
- 最终数字一律以 `evaluate_model.py` 输出为准(AP/AR/FROC + 框中心定位误差)
- 结果输出:`runs/train/<name>/`(训练曲线)与 `out/eval/`(FROC 图 + 逐图预测 json)
- 没装 torch 时先验证环境:`python src/evaluate_model.py --self-test`(不需要 ultralytics)
