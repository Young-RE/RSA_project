"""第4周任务:数据集统计 —— 样本量、每图 TD 数分布、中心点空间热力图、框尺寸核验。

用法:
    python src/dataset_stats.py

输出: 终端统计表 + out/stats/ 下的图。
统计结果直接是报告 I.B Dataset 章节的素材(成员B 负责整理成文)。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import plot_setup  # noqa: F401  中文字体设置
import matplotlib.pyplot as plt

from data_utils import DATA_ROOT, OUT_ROOT, label_path_for, list_ir_images, load_yolo_bboxes, load_yolo_centers


def collect_stats(split: str):
    """返回 (n_images, n_tds, per_image_counts, all_centers, all_box_wh)"""
    per_image, centers, box_wh = [], [], []
    for img in list_ir_images(split):
        boxes = load_yolo_bboxes(label_path_for(img, "bbox"))
        per_image.append(len(boxes))
        for _cls, cx, cy, w, h in boxes:
            box_wh.append((w, h))
        centers.extend(load_yolo_centers(label_path_for(img, "center")))
    return len(per_image), sum(per_image), per_image, centers, box_wh


def main() -> None:
    out = OUT_ROOT / "stats"
    out.mkdir(parents=True, exist_ok=True)

    # ---- 每个划分的统计表 ----
    all_centers = []
    print(f"{'划分':<6} {'图像数':>6} {'TD目标数':>8} {'平均TD/图':>10}")
    for split in ("train", "test"):
        n_img, n_td, per_img, centers, _box_wh = collect_stats(split)
        all_centers.extend(centers)
        print(f"{split:<6} {n_img:>6} {n_td:>8} {n_td / n_img:>10.3f}")

    # ---- 每图 TD 数分布 ----
    _, _, per_img, _, _ = collect_stats("train")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_img, bins=np.arange(-0.5, max(per_img) + 1.5, 1), color="#4C72B0")
    ax.set_xlabel("每张图的 TD 目标数")
    ax.set_ylabel("图像数量")
    ax.set_title("训练集每图 TD 数分布")
    fig.tight_layout()
    fig.savefig(out / "tds_per_image.png", dpi=150)
    print(f"已保存: {out / 'tds_per_image.png'}")

    # ---- 中心点空间热力图(相当于 TD 出现位置的分布图) ----
    centers = np.array(all_centers)
    fig, ax = plt.subplots(figsize=(6, 6))
    h = ax.hist2d(centers[:, 0], centers[:, 1], bins=64, range=[[0, 1], [0, 1]], cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="TD 数量")
    ax.set_xlabel("归一化 x(对应经度 109°E → 179°E)")
    ax.set_ylabel("归一化 y(对应纬度 15°S → 55°N)")
    ax.set_title("TD 中心点空间分布热力图(train+test)")
    fig.tight_layout()
    fig.savefig(out / "center_heatmap.png", dpi=150)
    print(f"已保存: {out / 'center_heatmap.png'}")

    # ---- 框尺寸核验(应为固定 5°×5° ≈ 45.5px ≈ 0.071 归一化) ----
    _, _, _, _, box_wh = collect_stats("train")
    wh = np.array(box_wh)
    print(f"bbox 尺寸: w 均值 {wh[:, 0].mean():.4f} ± {wh[:, 0].std():.4f}, "
          f"h 均值 {wh[:, 1].mean():.4f} ± {wh[:, 1].std():.4f} (归一化)")
    print("说明: 5°/0.11(°/px) = 45.5 px = 0.0711 归一化, 与标签吻合即为固定框。")


if __name__ == "__main__":
    main()
