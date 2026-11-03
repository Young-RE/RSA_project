"""成员A:准备 ultralytics 格式数据集 —— 按台风划分 train/val。

为什么按台风分:同一台风连续时刻(间隔 6h)的图像高度相似,
随机切分会让验证集里出现训练集"几乎一样"的图,导致指标虚高。
这里按台风名分组,把约 20% 的台风整体划入 val。这一点要写进报告 I.B。

用法:
    python src/prepare_yolo_dataset.py                    # 默认 20% 台风进 val
    python src/prepare_yolo_dataset.py --val-ratio 0.15 --seed 42

输出:
    out/yolo_ir/data.yaml            ultralytics 数据集配置(绝对路径,已生成好)
    out/yolo_ir/{train,val,test}/{images,labels}/
    终端打印各划分样本数与 val 台风名单
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

from data_utils import DATA_ROOT, OUT_ROOT, label_path_for, list_ir_images


def typhoon_key(stem: str) -> str:
    """NWP1601_NEPARTAK_3_... → NEPARTAK。多台风拼接图取第一个(主)台风。"""
    parts = stem.split("_")
    return parts[1] if len(parts) > 1 else stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-ratio", type=float, default=0.2, help="台风比例划入验证集")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_root = OUT_ROOT / "yolo_ir"
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "val", "test"):
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    # ---- 训练集按台风分 train/val ----
    train_imgs = list_ir_images("train")
    keys = sorted({typhoon_key(p.stem) for p in train_imgs})
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * args.val_ratio))
    val_keys = set(keys[:n_val])
    print(f"共 {len(keys)} 个台风, {n_val} 个划入验证集:")
    print("  val 台风: " + ", ".join(keys[:n_val]))

    counts = {"train": 0, "val": 0, "test": 0}
    multi = 0  # 多台风拼接图数量
    for img in train_imgs:
        dest = "val" if typhoon_key(img.stem) in val_keys else "train"
        _copy_pair(img, out_root / dest, counts)
        if img.stem.count("NWP") > 1:
            multi += 1
    for img in list_ir_images("test"):
        _copy_pair(img, out_root / "test", counts)

    print(f"样本划分: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print(f"其中多台风拼接图 {multi} 张(按主台风划分, 已尽量减小泄漏)")

    # ---- 生成 data.yaml(绝对路径, 每台机器各自运行本脚本生成) ----
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(
        f"path: {out_root.as_posix()}\n"
        f"train: train\nval: val\ntest: test\n"
        f"nc: 1\nnames:\n  0: TD\n",
        encoding="utf-8",
    )
    print(f"数据集配置: {yaml_path}")
    print("下一步: python src/train_yolo.py --model yolov8n.pt --epochs 100")


def _copy_pair(img: Path, dest_root: Path, counts: dict) -> None:
    """复制图像 + 对应 bbox 标签到 ultralytics 目录结构。"""
    label = label_path_for(img, "bbox")
    if not label.exists():
        print(f"[警告] 缺标签, 跳过: {img.name}")
        return
    shutil.copy2(img, dest_root / "images" / img.name)
    shutil.copy2(label, dest_root / "labels" / label.name)
    counts[dest_root.name] += 1


if __name__ == "__main__":
    main()
