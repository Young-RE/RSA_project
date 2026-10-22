"""统一评估指标 —— 全组所有模型必须用本模块计算最终指标(成员C 维护,全组合同)。

检测任务: AP / AR / FROC 曲线
定位任务: ME / MAE / RMSE / MD(同时给像素与度数两种单位)

口径约定(与报告模板一致):
- 预测框与真值框均为像素坐标 (x1, y1, x2, y2),预测需带 score
- IoU 阈值默认 0.5(对标说明文档 slide16 的 DINO 参考表)
- FROC: x 轴为平均每图假阳性数(FPI),y 轴为灵敏度(召回)
- 中心点距离换算: 0.11°/像素(data_utils.DEG_PER_PIX)

用法:
    python src/eval_metrics.py    # 跑合成数据自测,确认所有函数行为正确
"""
from __future__ import annotations

import numpy as np

from data_utils import DEG_PER_PIX


# ---------- 基础函数 ----------

def compute_iou(a, b) -> float:
    """两个 (x1, y1, x2, y2) 框的 IoU。"""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_detections(preds, gts, iou_thr: float):
    """贪心匹配:preds 按 score 降序,每个真值最多被匹配一次。

    参数:
        preds: [(x1, y1, x2, y2, score), ...]
        gts:   [(x1, y1, x2, y2), ...]
    返回:
        tp:    bool 数组,与按 score 降序排列后的 preds 对齐
        n_gt:  真值总数
    """
    order = np.argsort([-p[4] for p in preds])  # score 降序
    matched = np.zeros(len(gts), dtype=bool)
    tp = np.zeros(len(preds), dtype=bool)
    for i in order:
        best_j, best_iou = -1, iou_thr
        for j, gt in enumerate(gts):
            if matched[j]:
                continue
            iou = compute_iou(preds[i][:4], gt)
            if iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0:
            tp[i] = True
            matched[best_j] = True
    return tp, len(gts)


def compute_ap_ar(preds, gts, iou_thr: float = 0.5) -> dict:
    """单张图/全集的 AP 与 AR。

    AP 采用 all-point 插值(PR 曲线上对 recall 轴积分)。
    """
    tp, n_gt = match_detections(preds, gts, iou_thr)
    scores = np.array([p[4] for p in preds])[np.argsort([-p[4] for p in preds])]
    tp = tp[np.argsort([-p[4] for p in preds])]
    cum_tp = np.cumsum(tp)
    n_pred = len(preds)
    precisions = cum_tp / np.arange(1, n_pred + 1)
    recalls = cum_tp / n_gt if n_gt else np.zeros(n_pred)

    # all-point AP:对 recall 单调化后的 PR 曲线下面积
    mprec = np.maximum.accumulate(precisions[::-1])[::-1]
    mrec = np.concatenate(([0.0], recalls, [recalls[-1]]))
    mprec = np.concatenate(([0.0], mprec, [0.0]))
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mprec[1:]))
    ar = float(recalls[-1]) if n_pred else 0.0
    return {"ap": ap, "ar": ar, "n_gt": n_gt, "n_pred": n_pred,
            "precisions": precisions, "recalls": recalls, "scores": scores}


def froc_curve(preds_per_image, gts_per_image, iou_thr: float = 0.5,
               max_fpi: float = 8.0, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """全测试集的 FROC 曲线。

    参数:
        preds_per_image: list[list[(x1,y1,x2,y2,score)]] 每张图的预测
        gts_per_image:   list[list[(x1,y1,x2,y2)]]       每张图的真值
    返回:
        (fpis, recalls): 在 [0, max_fpi] 上均匀采样的 FPI 与对应召回
    """
    # 全图所有预测按 score 降序;记录每个预测属于哪张图
    all_preds, img_ids = [], []
    for i, preds in enumerate(preds_per_image):
        for p in preds:
            all_preds.append(p)
            img_ids.append(i)
    all_preds = np.array(all_preds, dtype=object) if all_preds else np.empty((0, 5))
    img_ids = np.array(img_ids, dtype=int)
    n_img = max(len(gts_per_image), 1)
    n_gt_total = sum(len(g) for g in gts_per_image)

    if len(all_preds) == 0:
        xs = np.linspace(0, max_fpi, n_points)
        return xs, np.zeros_like(xs)

    order = np.argsort(-all_preds[:, 4].astype(float))
    fp_count = np.zeros(n_img, dtype=int)
    tp_count = 0
    curve = []  # (fpi, recall)
    # 每图内做贪心匹配状态
    matched = [np.zeros(len(g), dtype=bool) for g in gts_per_image]
    for i in order:
        img_i = img_ids[i]
        iou_best, j_best = iou_thr, -1
        for j, gt in enumerate(gts_per_image[img_i]):
            if matched[img_i][j]:
                continue
            iou = compute_iou(all_preds[i][:4].astype(float), gt)
            if iou > iou_best:
                iou_best, j_best = iou, j
        if j_best >= 0:
            tp_count += 1
            matched[img_i][j_best] = True
        else:
            fp_count[img_i] += 1
        curve.append((fp_count.sum() / n_img, tp_count / n_gt_total if n_gt_total else 0.0))

    # 对 FPI 轴插值(step 曲线,取每个 FPI 处已见到的最大 recall)
    curve = np.array(curve)
    xs = np.linspace(0, max_fpi, n_points)
    ys = np.zeros_like(xs)
    for k, x in enumerate(xs):
        mask = curve[:, 0] <= x
        ys[k] = curve[mask, 1].max() if mask.any() else 0.0
    return xs, ys


# ---------- 中心点定位 ----------

def center_errors(pred_centers, gt_centers, deg_per_pix: float = DEG_PER_PIX) -> dict:
    """中心点定位误差。pred/gt 为像素坐标 [(x, y), ...]。

    数量不一致时按最近邻贪心配对(定位任务中 pred 与 gt 应一一对应,
    若模型输出多于真值,多余预测不计入)。返回像素与度数两种单位。
    """
    pred_centers = [tuple(map(float, p)) for p in pred_centers]
    gt_centers = [tuple(map(float, g)) for g in gt_centers]
    used = np.zeros(len(gt_centers), dtype=bool)
    errs = []
    for p in pred_centers:
        best_j, best_d = -1, np.inf
        for j, g in enumerate(gt_centers):
            if used[j]:
                continue
            d = np.hypot(p[0] - g[0], p[1] - g[1])
            if d < best_d:
                best_j, best_d = j, d
        if best_j >= 0:
            used[best_j] = True
            errs.append(best_d)
    if not errs:
        return {"me": float("nan"), "mae": float("nan"), "rmse": float("nan"),
                "md": float("nan"), "n": 0}
    e = np.array(errs)
    out = {
        "me": float(e.mean()),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "md": float(np.median(e)),
        "n": len(e),
    }
    # 度数单位副本
    out.update({f"{k}_deg": v * deg_per_pix for k, v in out.items()
                if k in ("me", "mae", "rmse", "md")})
    return out


def centers_to_heatmap(centers, size: int = 640, sigma: float = 3.0) -> np.ndarray:
    """中心点(像素坐标)→ 高斯热图 (size, size)。成员C 的 heatmap 回归用。"""
    heat = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for cx, cy in centers:
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        heat = np.maximum(heat, g)
    return heat


# ---------- 自测 ----------

if __name__ == "__main__":
    print("eval_metrics 自测(合成数据):")
    # 完美预测
    gts = [(10, 10, 55, 55), (100, 100, 145, 145)]
    preds = [(10, 10, 55, 55, 0.9), (100, 100, 145, 145, 0.8)]
    r = compute_ap_ar(preds, gts)
    print(f"  完美预测: AP={r['ap']:.3f} AR={r['ar']:.3f} (应均为 1.000)")
    assert abs(r["ap"] - 1) < 1e-6 and abs(r["ar"] - 1) < 1e-6

    # 一半命中 + 一个误报
    preds2 = [(10, 10, 55, 55, 0.9), (300, 300, 320, 320, 0.8)]
    r2 = compute_ap_ar(preds2, gts)
    print(f"  1命中1误报: AP={r2['ap']:.3f} AR={r2['ar']:.3f} (AR 应为 0.500)")
    assert abs(r2["ar"] - 0.5) < 1e-6

    # FROC
    fpis, recalls = froc_curve([preds2], [gts], max_fpi=2)
    print(f"  FROC: fpi={fpis[[0, -1]]}, recall={recalls[[0, -1]]}")

    # 中心点误差: 误差为 0 和 sqrt(2), MAE = sqrt(2)/2 ≈ 0.7071
    c = center_errors([(50, 50), (101, 101)], [(50, 50), (100, 100)])
    print(f"  中心点: MAE={c['mae']:.2f}px = {c['mae_deg']:.2f}°, RMSE={c['rmse']:.2f}px, MD={c['md']:.2f}px")
    assert abs(c["mae"] - np.sqrt(2) / 2) < 1e-6
    print("  (自测通过)")
