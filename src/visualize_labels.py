"""第4周任务:单帧红外图可视化 —— 叠加 bbox + 中心点标注。

用法:
    python src/visualize_labels.py                      # 训练集第 0 张
    python src/visualize_labels.py --split test --index 5

输出: out/vis/<文件名>.png
每位组员都要跑通本脚本,把输出图发到群里(验收:能看懂图里画的是什么)。
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import plot_setup  # noqa: F401  中文字体设置
import matplotlib.pyplot as plt

from data_utils import (DATA_ROOT, OUT_ROOT, SIZE, label_path_for, list_ir_images,
                        load_ir_image, load_yolo_bboxes, load_yolo_centers,
                        yolo_to_pixel)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--index", type=int, default=0, help="该划分下第几张(按文件名排序)")
    args = ap.parse_args()

    imgs = list_ir_images(args.split)
    if args.index >= len(imgs):
        sys.exit(f"index 越界: {args.split} 只有 {len(imgs)} 张图")
    img_path = imgs[args.index]

    img = load_ir_image(img_path)
    boxes = load_yolo_bboxes(label_path_for(img_path, "bbox"))
    centers = load_yolo_centers(label_path_for(img_path, "center"))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    for _cls, cx, cy, w, h in boxes:
        x1, y1, x2, y2 = yolo_to_pixel(cx, cy, w, h)
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="red", lw=1.5))
    for cx, cy in centers:
        ax.plot(cx * SIZE, cy * SIZE, "r+", ms=14, mew=2)
    ax.set_title(f"{img_path.name}\n{args.split} #{args.index}: {len(boxes)} 个 TD 框, {len(centers)} 个中心点")
    ax.set_xlabel("像素 x")
    ax.set_ylabel("像素 y")

    out = OUT_ROOT / "vis"
    out.mkdir(parents=True, exist_ok=True)
    save = out / f"{img_path.stem}.png"
    fig.savefig(save, dpi=150, bbox_inches="tight")
    print(f"已保存: {save}")
    print(f"样本: {len(boxes)} 个 bbox | {len(centers)} 个中心点 | 图像 {img.shape} 值域 [{img.min():.3f}, {img.max():.3f}]")


if __name__ == "__main__":
    main()
