"""自定义模块注册 —— 让 ultralytics 能解析含 CBAM 的模型 yaml。

原理: ultralytics.nn.tasks.parse_model 在模块全局命名空间里按名字查找层类型,
本文件把 CBAM 注入该命名空间后, yaml 里写 `- [-1, 1, CBAM, [channels]]` 即可。

用法: 训练脚本 import 本文件(见 train_yolo.py, 已自动导入)。
"""
import torch
from torch import nn


class ChannelAttention(nn.Module):
    """通道注意力: 平均池化 + 最大池化共享一个 MLP, 输出逐通道权重。"""

    def __init__(self, channels: int, ratio: int = 16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // ratio, channels, 1, bias=False),
        )

    def forward(self, x):
        w = torch.sigmoid(self.fc(self.avg(x)) + self.fc(self.max(x)))
        return x * w


class SpatialAttention(nn.Module):
    """空间注意力: 沿通道取均值/最大值拼成 2 通道图, 卷积出逐位置权重。"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """Convolutional Block Attention Module: 通道注意力 + 空间注意力, 不改变特征形状。"""

    def __init__(self, channels: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


def register() -> None:
    """把自定义层注入 ultralytics 的解析命名空间。"""
    import ultralytics.nn.tasks as tasks
    tasks.CBAM = CBAM
    tasks.ChannelAttention = ChannelAttention
    tasks.SpatialAttention = SpatialAttention


register()
