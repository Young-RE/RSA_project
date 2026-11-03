"""成员A:训练曲线可视化 —— 读 ultralytics results.csv, 画 loss 与 mAP50 曲线。

所有模型通用(成员B/D 的训练记录也能用), 输出报告 E 章的曲线素材。

用法:
    python src/plot_training_curves.py                        # 画最近的实验
    python src/plot_training_curves.py --name yolov8n_e100    # 指定实验目录名

输出: runs/train/<name>/curves.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import plot_setup  # noqa: F401  中文字体设置
import matplotlib.pyplot as plt

from data_utils import PROJECT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default=None, help="runs/train/ 下的实验目录名, 默认取最新的")
    args = ap.parse_args()

    run_dir = PROJECT_ROOT / "runs" / "train"
    if args.name:
        exp = run_dir / args.name
    else:
        exp = max([d for d in run_dir.iterdir() if (d / "results.csv").exists()],
                  key=lambda d: d.stat().st_mtime)
    csv_path = exp / "results.csv"
    if not csv_path.exists():
        raise SystemExit(f"找不到 {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    best_ep = int(df["metrics/mAP50(B)"].idxmax()) + 1
    best_map = df["metrics/mAP50(B)"].max()
    final_map = df["metrics/mAP50(B)"].iloc[-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.plot(df["epoch"], df["train/box_loss"], label="train box_loss", lw=1.6)
    ax.plot(df["epoch"], df["val/box_loss"], label="val box_loss", lw=1.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("box_loss 曲线"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="val mAP50", lw=1.6, color="#4C72B0")
    ax.axvline(best_ep, color="r", ls="--", lw=1, label=f"best epoch {best_ep}")
    ax.set_xlabel("epoch"); ax.set_ylabel("mAP50")
    ax.set_title(f"验证集 mAP50(最优 {best_map:.4f} @ ep{best_ep}, 末轮 {final_map:.4f})")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle(exp.name)
    fig.tight_layout()
    out = exp / "curves.png"
    fig.savefig(out, dpi=150)
    print(f"已保存: {out}")
    print(f"最优 mAP50={best_map:.4f} @ epoch {best_ep} | 末轮 mAP50={final_map:.4f}")


if __name__ == "__main__":
    main()
