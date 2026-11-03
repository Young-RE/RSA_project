"""成员A:统一评估 —— 用 eval_metrics 口径评估训练好的检测模型。

统一口径(全组合同, 最终数字一律来自本脚本):
- 检测: AP@IoU=0.5、AR、FROC 曲线
- 定位: ME/MAE/RMSE/MD(像素与度数)—— 检测框中心 vs TD_center 标签

用法:
    python src/evaluate_model.py --weights runs/train/yolov8n_e100/weights/best.pt
    python src/evaluate_model.py --self-test   # 无 ultralytics 依赖, 验证评估管线

输出:
    out/eval/<实验名>_froc.png    FROC 曲线(报告 E 章素材)
    out/eval/<实验名>_preds.json 逐图预测(供成员C 汇总对比总表)
    终端打印 AP/AR + 中心点误差表
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import plot_setup  # noqa: F401  中文字体设置
import matplotlib.pyplot as plt

from data_utils import (DATA_ROOT, DEG_PER_PIX, OUT_ROOT, SIZE, load_yolo_bboxes,
                        load_yolo_centers, yolo_to_pixel)
from eval_metrics import center_errors, evaluate_detection_dataset

IOU_THR = 0.5      # 统一 IoU 阈值(对标说明文档 slide16 DINO 参考表)
CONF_THR = 0.001   # 推断保留几乎全部候选; AP 按分数排序计算, 不卡置信度阈值


def aggregate_center_errors(errs_px: list[float]) -> dict:
    """逐图收集的像素误差 → 汇总指标(像素 + 度数)。"""
    if not errs_px:
        return {"n": 0}
    e = np.array(errs_px)
    out = {"n": len(e),
           "me": float(e.mean()),
           "mae": float(np.abs(e).mean()),
           "rmse": float(np.sqrt((e ** 2).mean())),
           "md": float(np.median(e))}
    out.update({f"{k}_deg": v * DEG_PER_PIX for k, v in out.items()
                if k in ("me", "mae", "rmse", "md")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=None, help="训练好的 .pt 权重")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.weights:
        sys.exit("需要 --weights <路径> 或 --self-test")

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("未安装 ultralytics。先装深度学习环境(README.md),再运行本脚本。")

    img_dir = OUT_ROOT / "yolo_ir" / args.split / "images"
    imgs = sorted(img_dir.glob("*.png"))
    if not imgs:
        sys.exit(f"没有图像: {img_dir}\n请先运行: python src/prepare_yolo_dataset.py")

    # ---- 推理(按文件名索引, 与真值顺序对齐) ----
    model = YOLO(args.weights)
    results = model.predict(source=str(img_dir), conf=CONF_THR, iou=0.65,
                            imgsz=SIZE, verbose=False)
    preds_by_stem = {}
    for r in results:
        preds = []
        if r.boxes is not None:
            for row in r.boxes.data.tolist():  # x1,y1,x2,y2,conf,cls
                x1, y1, x2, y2, conf = row[:5]
                preds.append((float(x1), float(y1), float(x2), float(y2), float(conf)))
        preds_by_stem[Path(r.path).stem] = preds

    # ---- 真值与逐图预测 ----
    preds_per_image, gts_per_image = [], []
    pred_centers_per_image, gt_centers_per_image = [], []
    for img in imgs:
        lbl = OUT_ROOT / "yolo_ir" / args.split / "labels" / (img.stem + ".txt")
        boxes = load_yolo_bboxes(lbl) if lbl.exists() else []
        gts_per_image.append([yolo_to_pixel(cx, cy, w, h) for _c, cx, cy, w, h in boxes])

        preds = preds_by_stem.get(img.stem, [])
        preds_per_image.append(preds)
        pred_centers_per_image.append([((x1 + x2) / 2, (y1 + y2) / 2)
                                       for x1, y1, x2, y2, _s in preds])
        clbl = (DATA_ROOT / "single_frame" / "ir" / args.split
                / "labels" / "TD_center" / "yolo" / (img.stem + ".txt"))
        cts = load_yolo_centers(clbl) if clbl.exists() else []
        gt_centers_per_image.append([(cx * SIZE, cy * SIZE) for cx, cy in cts])

    # ---- 统一口径评估 ----
    det = evaluate_detection_dataset(preds_per_image, gts_per_image, IOU_THR)
    errs_px = []
    for pc, gc in zip(pred_centers_per_image, gt_centers_per_image):
        errs_px.extend(center_errors(pc, gc)["errs"])
    ctr = aggregate_center_errors(errs_px)

    print(f"\n===== 检测({args.split} 集, IoU={IOU_THR}) =====")
    print(f"AP = {det['ap']:.4f}   AR = {det['ar']:.4f}")
    print(f"真值 {det['n_gt']} 个, 预测 {det['n_pred']} 个")
    print(f"===== 中心点定位(框中心 vs TD_center) =====")
    if ctr.get("n"):
        print(f"ME={ctr['me']:.2f}px({ctr['me_deg']:.2f}°)  MAE={ctr['mae']:.2f}px({ctr['mae_deg']:.2f}°)  "
              f"RMSE={ctr['rmse']:.2f}px({ctr['rmse_deg']:.2f}°)  MD={ctr['md']:.2f}px({ctr['md_deg']:.2f}°)  n={ctr['n']}")
    else:
        print("无有效配对(预测为空)")

    # ---- 保存结果 ----
    out = OUT_ROOT / "eval"
    out.mkdir(parents=True, exist_ok=True)
    exp_name = Path(args.weights).parent.parent.name  # runs/train/<name>/weights/best.pt
    tag = f"{exp_name}_{args.split}"

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(det["fpis"], det["froc_recalls"], lw=2, color="#4C72B0")
    ax.set_xlabel("平均每图假阳性数 FPI")
    ax.set_ylabel("灵敏度 Recall")
    ax.set_title(f"FROC 曲线 —— {tag}\nAP@IoU{IOU_THR} = {det['ap']:.4f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{tag}_froc.png", dpi=150)
    print(f"已保存: {out / (tag + '_froc.png')}")

    preds_json = {img.stem: [{"box": list(map(float, b[:4])), "score": float(b[4])}
                             for b in preds_by_stem.get(img.stem, [])] for img in imgs}
    json_path = out / f"{tag}_preds.json"
    json_path.write_text(json.dumps(preds_json, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已保存: {json_path}(逐图预测, 供成员C 汇总)")


def self_test() -> None:
    """合成数据验证评估管线(不需要 ultralytics)。"""
    gts = [[(10, 10, 55, 55), (100, 100, 145, 145)]]
    preds = [[(10, 10, 55, 55, 0.9), (300, 300, 320, 320, 0.8)]]
    det = evaluate_detection_dataset(preds, gts, IOU_THR)
    print(f"自测: AP={det['ap']:.3f} AR={det['ar']:.3f} (AR 应为 0.500)")
    c = center_errors([(32.5, 32.5)], [(32.5, 32.5)])
    print(f"自测: 中心点 MAE={c['mae']:.3f}px (应为 0.000)")
    assert abs(det["ar"] - 0.5) < 1e-6 and c["mae"] < 1e-6
    print("评估管线自测通过 [OK]")


if __name__ == "__main__":
    main()
