"""第4周任务(成员D 必跑):多帧序列可视化 —— 读一个 (7,640,640) .npy 序列,输出动图。

用法:
    python src/view_multiframe.py --hours 6 --split train --index 0

输出: out/vis/<文件名>.gif 与 <文件名>_grid.png(7 帧平铺)
多帧序列时间排列为 [T, T-Δ, T-2Δ, ..., T-6Δ],Δ = 6h/12h/18h 中的间隔。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import plot_setup  # noqa: F401  中文字体设置
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from data_utils import DATA_ROOT, OUT_ROOT, label_path_for, load_yolo_centers


def list_sequences(hours: int, split: str) -> list[Path]:
    return sorted((DATA_ROOT / f"Multiple_frames_{hours}h" / split / "images").glob("*.npy"))


def norm_for_display(frame: np.ndarray) -> np.ndarray:
    """单帧显示归一化:按 2/98 分位数拉伸到 [0,1],提高对比度。"""
    lo, hi = np.percentile(frame, 2), np.percentile(frame, 98)
    if hi - lo < 1e-8:
        hi = lo + 1e-8
    return np.clip((frame - lo) / (hi - lo), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=int, default=6, choices=[6, 12, 18])
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    seqs = list_sequences(args.hours, args.split)
    if args.index >= len(seqs):
        sys.exit(f"index 越界: {args.split} 只有 {len(seqs)} 个序列")
    seq_path = seqs[args.index]
    seq = np.load(seq_path)
    print(f"序列形状: {seq.shape}, dtype: {seq.dtype}, 值域 [{seq.min():.3f}, {seq.max():.3f}]")

    n = seq.shape[0]
    frames = [norm_for_display(seq[i]) for i in range(n)]

    # 尝试读取 T 时刻中心点标签(若存在)
    centers = []
    center_path = label_path_for(seq_path, "center")
    if center_path.exists():
        centers = load_yolo_centers(center_path)

    out = OUT_ROOT / "vis"
    out.mkdir(parents=True, exist_ok=True)

    # ---- gif 动图(matplotlib PillowWriter,无需 imageio) ----
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    im = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=1)
    for cx, cy in centers:
        ax.plot(cx * seq.shape[2], cy * seq.shape[1], "r+", ms=12, mew=2)
    ax.axis("off")

    def update(i: int):
        im.set_data(frames[i])
        ax.set_title("T" if i == 0 else f"T-{i}Δ", fontsize=11)
        return [im]

    gif_path = out / f"{seq_path.stem}.gif"
    ani = animation.FuncAnimation(fig, update, frames=n, interval=400, blit=True)
    ani.save(gif_path, writer=animation.PillowWriter(fps=2.5))
    plt.close(fig)
    print(f"已保存动图: {gif_path}")

    # ---- 7 帧平铺图 ----
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n):
        ax = axes[i]
        ax.imshow(frames[i], cmap="gray", vmin=0, vmax=1)
        if centers and i == 0:
            for cx, cy in centers:
                ax.plot(cx * seq.shape[2], cy * seq.shape[1], "r+", ms=12, mew=2)
        ax.set_title("T" if i == 0 else f"T-{i}Δ")
        ax.axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"{seq_path.name}(红+ 为 T 时刻中心点标签)", fontsize=9)
    fig.tight_layout()
    grid_path = out / f"{seq_path.stem}_grid.png"
    fig.savefig(grid_path, dpi=130, bbox_inches="tight")
    print(f"已保存平铺图: {grid_path}")
    print("说明: 第 0 帧为 T 时刻(有标签), 之后为 T-Δ, T-2Δ, ..., 共 7 帧")


if __name__ == "__main__":
    main()
