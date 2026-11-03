"""多模型对比汇总 —— 读 out/eval/ 下所有逐图预测 json, 用统一口径出对比表与合并 FROC 图。

(成员C 的对比总表工具; 任何模型只要经过 evaluate_model.py 评估, 自动进入汇总。)

用法:
    python src/compare_models.py [--center-conf 0.25]

输出:
    out/eval/comparison_froc.png   所有模型 FROC 曲线合图
    out/eval/comparison_table.md   报告可直接粘贴的对比表
    终端打印 AP/AR/中心点误差对比
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import plot_setup  # noqa: F401  中文字体设置
import matplotlib.pyplot as plt

from data_utils import (DATA_ROOT, DEG_PER_PIX, OUT_ROOT, SIZE, load_yolo_bboxes,
                        load_yolo_centers, yolo_to_pixel)
from eval_metrics import center_errors, evaluate_detection_dataset

IOU_THR = 0.5
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]


def load_gt(split: str):
    """从准备好的数据集读真值框与中心点(像素坐标)。"""
    gts, centers = [], []
    lbl_dir = OUT_ROOT / "yolo_ir" / split / "labels"
    c_dir = DATA_ROOT / "single_frame" / "ir" / split / "labels" / "TD_center" / "yolo"
    for lbl in sorted(lbl_dir.glob("*.txt")):
        boxes = load_yolo_bboxes(lbl)
        gts.append([yolo_to_pixel(cx, cy, w, h) for _c, cx, cy, w, h in boxes])
        clbl = c_dir / lbl.name
        cts = load_yolo_centers(clbl) if clbl.exists() else []
        centers.append([(cx * SIZE, cy * SIZE) for cx, cy in cts])
    return gts, centers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--center-conf", type=float, default=0.25,
                    help="中心点定位使用的高置信度阈值, 与 evaluate_model.py 保持一致")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    gts_per_image, gt_centers_per_image = load_gt(args.split)

    pred_files = sorted((OUT_ROOT / "eval").glob(f"*_{args.split}_preds.json"))
    if not pred_files:
        raise SystemExit("out/eval/ 下没有预测文件, 先跑 evaluate_model.py")

    rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, pf in enumerate(pred_files):
        name = pf.stem.replace(f"_{args.split}_preds", "")
        preds_by_stem = json.loads(pf.read_text(encoding="utf-8"))
        preds_per_image, pred_centers_per_image = [], []
        for lbl in sorted((OUT_ROOT / "yolo_ir" / args.split / "labels").glob("*.txt")):
            preds = [tuple(map(float, b["box"])) + (float(b["score"]),)
                     for b in preds_by_stem.get(lbl.stem, [])]
            preds_per_image.append(preds)
            pred_centers_per_image.append(
                [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2, s in preds if s >= args.center_conf])

        det = evaluate_detection_dataset(preds_per_image, gts_per_image, IOU_THR)
        errs = []
        for pc, gc in zip(pred_centers_per_image, gt_centers_per_image):
            errs.extend(center_errors(pc, gc)["errs"])
        md = float(np.median(errs) * DEG_PER_PIX) if errs else float("nan")
        mae = float(np.mean(errs) * DEG_PER_PIX) if errs else float("nan")

        rows.append((name, det["ap"], det["ar"], mae, md, len(errs)))
        ax.plot(det["fpis"], det["froc_recalls"], lw=1.8, color=COLORS[i % len(COLORS)],
                label=f"{name} (AP={det['ap']:.4f})")

    # ---- 输出 ----
    out = OUT_ROOT / "eval"
    ax.set_xlabel("平均每图假阳性数 FPI")
    ax.set_ylabel("灵敏度 Recall")
    ax.set_title(f"FROC 曲线对比({args.split} 集, IoU={IOU_THR})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "comparison_froc.png", dpi=150)
    print(f"已保存: {out / 'comparison_froc.png'}")

    rows.sort(key=lambda r: -r[1])
    print(f"\n{'模型':<28} {'AP':>7} {'AR':>7} {'MAE°':>7} {'MD°':>7} {'配对n':>5}")
    for name, ap, ar, mae, md, n in rows:
        print(f"{name:<28} {ap:>7.4f} {ar:>7.4f} {mae:>7.2f} {md:>7.2f} {n:>5}")

    lines = ["| 模型 | AP@IoU0.5 | AR | 中心点 MAE(°) | 中心点 MD(°) |",
             "|---|---|---|---|---|"]
    for name, ap, ar, mae, md, _n in rows:
        lines.append(f"| {name} | {ap:.4f} | {ar:.4f} | {mae:.2f} | {md:.2f} |")
    md_path = out / "comparison_table.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已保存: {md_path}(报告可直接粘贴)")


if __name__ == "__main__":
    main()
