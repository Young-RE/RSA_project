"""数据读取工具 —— 全组共用。

路径约定:
- 项目根目录为 src/ 的上一级
- 单帧红外图像: dataset/single_frame/ir/{split}/images/*.png (8-bit 灰度 640x640)
- bbox 标签(YOLO 格式):   dataset/single_frame/ir/{split}/labels/TD_bbox/yolo/<同名>.txt
  每行: class cx cy w h(归一化坐标)
- center 标签:           dataset/single_frame/ir/{split}/labels/TD_center/yolo/<同名>.txt
  每行: cx cy(归一化坐标)
- 多帧序列: dataset/Multiple_frames_{6h|12h|18h}/{split}/images/*.npy
  形状 (7, 640, 640) float32, 时间排列 [T, T-Δ, T-2Δ, ..., T-6Δ]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "dataset"
OUT_ROOT = PROJECT_ROOT / "out"

SIZE = 640          # 图像像素尺寸
DEG_PER_PIX = 0.11  # 每像素对应的经纬度数(空间分辨率)


# ---------- 单帧红外 ----------

def list_ir_images(split: str = "train") -> list[Path]:
    """返回某个划分(train/test)下的所有红外图像路径。"""
    return sorted((DATA_ROOT / "single_frame" / "ir" / split / "images").glob("*.png"))


def load_ir_image(path: Path) -> np.ndarray:
    """读取红外 PNG,返回 float32 数组 (640, 640),值域 [0, 1]。"""
    img = np.asarray(Image.open(path), dtype=np.float32)
    if img.ndim == 3:  # 万一是 RGB,取第一通道
        img = img[..., 0]
    return img / 255.0


def load_yolo_bboxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    """读取 YOLO bbox 标签,返回 [(class, cx, cy, w, h)],归一化坐标。"""
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 5:
            cls, cx, cy, w, h = int(float(parts[0])), *(float(p) for p in parts[1:])
        elif len(parts) == 4:  # 无类别号,按 bbox 读
            cls, cx, cy, w, h = 0, *(float(p) for p in parts)
        else:
            raise ValueError(f"无法解析标签行: {line!r} in {path}")
        boxes.append((cls, cx, cy, w, h))
    return boxes


def load_yolo_centers(path: Path) -> list[tuple[float, float]]:
    """读取中心点标签,返回 [(cx, cy)],归一化坐标。bbox 样式行也兼容。"""
    centers = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 5:  # bbox 样式 → 取中心
            centers.append((float(parts[1]), float(parts[2])))
        elif len(parts) == 2:
            centers.append((float(parts[0]), float(parts[1])))
        else:
            raise ValueError(f"无法解析标签行: {line!r} in {path}")
    return centers


def label_path_for(img_path: Path, kind: str = "bbox") -> Path:
    """由图像路径得到标签路径。kind: 'bbox' | 'center'。

    支持单帧 png 与多帧 npy(多帧标签位于同一 labels 目录树下)。
    """
    label_dir = img_path.parent.parent / "labels"
    return label_dir / f"TD_{kind}" / "yolo" / (img_path.stem + ".txt")


# ---------- 坐标换算 ----------

def yolo_to_pixel(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    """归一化 YOLO 框 → 像素坐标 (x1, y1, x2, y2)。"""
    x1 = (cx - w / 2) * SIZE
    y1 = (cy - h / 2) * SIZE
    x2 = (cx + w / 2) * SIZE
    y2 = (cy + h / 2) * SIZE
    return x1, y1, x2, y2


def deg_to_pixel(deg: float) -> float:
    """度数距离 → 像素距离(用于 3° 检测阈值等口径换算)。"""
    return deg / DEG_PER_PIX
