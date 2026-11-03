"""成员A:YOLO 训练入口(全组统一训练框架的第一条线)。

支持 YOLOv8 / YOLOv11 / 自定义模型 yaml(改进版用同一个入口)。

用法:
    python src/prepare_yolo_dataset.py                        # 先准备数据(每台机器一次)
    python src/train_yolo.py --model yolov8n.pt --epochs 100  # 基线①
    python src/train_yolo.py --model yolo11n.pt --epochs 100  # 对比
    python src/train_yolo.py --model <改进模型.yaml> --epochs 100

输出:
    runs/train/<name>/weights/best.pt
    runs/train/<name>/ 下的训练曲线与验证结果(报告素材)

训练完成后必须用统一口径评估:
    python src/evaluate_model.py --weights runs/train/<name>/weights/best.pt
"""
import argparse
import sys
from pathlib import Path

from data_utils import OUT_ROOT, PROJECT_ROOT, SIZE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolov8n.pt",
                    help="预训练权重(yolov8n.pt/yolo11n.pt)或自定义模型 yaml 路径")
    ap.add_argument("--data", default=None, help="默认 out/yolo_ir/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=SIZE)
    ap.add_argument("--device", default="", help="0/1/cpu, 留空自动选择")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader 进程数; Windows 上建议 0, Linux 服务器可设 8")
    ap.add_argument("--name", default=None, help="实验名, 默认 <模型名>_e<epochs>")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("未安装 ultralytics/torch。先装深度学习环境(README.md 第5周部分),再运行本脚本。")
    import custom_modules  # noqa: F401  注册 CBAM 等自定义层(自定义 yaml 需要)

    data = args.data or str(OUT_ROOT / "yolo_ir" / "data.yaml")
    if not Path(data).exists():
        sys.exit(f"数据集配置不存在: {data}\n请先运行: python src/prepare_yolo_dataset.py")

    name = args.name or f"{Path(args.model).stem}_e{args.epochs}"
    model = YOLO(args.model)
    model.train(
        data=data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device or None,
        workers=args.workers,
        seed=args.seed,
        name=name,
        project=str(PROJECT_ROOT / "runs" / "train"),
        val=True,
        plots=True,
    )
    best = PROJECT_ROOT / "runs" / "train" / name / "weights" / "best.pt"
    print(f"训练完成, 最佳权重: {best}")
    print(f"统一口径评估: python src/evaluate_model.py --weights {best.as_posix()}")


if __name__ == "__main__":
    main()
