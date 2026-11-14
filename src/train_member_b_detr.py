"""成员 B：用统一的单帧 IR 划分训练 DINO / RT-DETR，导出逐图预测。

先运行 prepare_yolo_dataset.py，再运行本脚本。DINO 需要 MMDetection 3.3
及其官方配置和 COCO 权重，RT-DETR 使用 Ultralytics 的 rtdetr-l.pt。
"""
import argparse
import json
from pathlib import Path

from data_utils import OUT_ROOT, PROJECT_ROOT, SIZE, load_yolo_bboxes, yolo_to_pixel
from evaluate_model import CONF_THR


IR = OUT_ROOT / "yolo_ir"


def write_coco(split: str, destination: Path) -> Path:
    images, annotations = [], []
    for image_id, image in enumerate(sorted((IR / split / "images").glob("*.png")), 1):
        images.append(dict(id=image_id, file_name=image.name, width=SIZE, height=SIZE))
        label = IR / split / "labels" / f"{image.stem}.txt"
        for _, cx, cy, width, height in load_yolo_bboxes(label):
            x1, y1, x2, y2 = yolo_to_pixel(cx, cy, width, height)
            w, h = min(SIZE, x2) - max(0, x1), min(SIZE, y2) - max(0, y1)
            if w <= 0 or h <= 0:
                raise ValueError(f"无效的边界框: {label}")
            annotations.append(dict(id=len(annotations) + 1, image_id=image_id,
                                    category_id=1, bbox=[max(0, x1), max(0, y1), w, h],
                                    area=w * h, iscrowd=0))
    if not images or not annotations:
        raise ValueError(f"{split} 缺少图像或检测框标签")
    target = destination / f"{split}.json"
    target.write_text(json.dumps(dict(images=images, annotations=annotations,
                                      categories=[dict(id=1, name="TD")])), encoding="utf-8")
    return target


def train_dino(args, run_dir: Path) -> Path:
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.dino_config)
    cfg.work_dir = str(run_dir)
    cfg.load_from = str(args.weights.resolve())
    cfg.model.bbox_head.num_classes = 1
    cfg.model.backbone.init_cfg = None
    cfg.randomness = dict(seed=42)
    cfg.env_cfg.mp_cfg.mp_start_method = "spawn"
    cfg.train_cfg.max_epochs = args.epochs
    cfg.param_scheduler[0].end = args.epochs
    cfg.param_scheduler[0].milestones = [max(1, args.epochs - 1)]
    cfg.optim_wrapper.type = "AmpOptimWrapper"
    cfg.default_hooks.checkpoint.update(save_best="coco/bbox_mAP_50", max_keep_ckpts=2)
    for split, loader in (("train", cfg.train_dataloader), ("val", cfg.val_dataloader)):
        ann_file = write_coco(split, run_dir)
        loader.num_workers = 0
        loader.persistent_workers = False
        loader.batch_size = args.batch if split == "train" else 1
        loader.dataset.update(data_root=str(IR), ann_file=str(ann_file),
                              data_prefix=dict(img=f"{split}/images"),
                              metainfo=dict(classes=("TD",)))
        pipeline = [dict(type="LoadImageFromFile"), dict(type="LoadAnnotations", with_bbox=True),
                    dict(type="Resize", scale=(SIZE, SIZE), keep_ratio=True)]
        if split == "train":
            pipeline.append(dict(type="RandomFlip", prob=0.5))
        loader.dataset.pipeline = [*pipeline, dict(type="PackDetInputs")]
        if split == "val":
            cfg.val_evaluator.ann_file = str(ann_file)
    Runner.from_cfg(cfg).train()
    return next(run_dir.glob("best_coco_bbox_mAP_50_epoch_*.pth"))


def predict_dino(args, run_dir: Path, checkpoint: Path, images: list[Path]) -> dict:
    from mmcv.transforms import Compose
    from mmdet.apis import inference_detector, init_detector

    config = run_dir / args.dino_config.name
    model = init_detector(str(config), str(checkpoint), device="cuda:0")
    pipeline = Compose([
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=(SIZE, SIZE), keep_ratio=True),
        dict(type="mmdet.PackDetInputs"),
    ])
    predictions = {}
    for image in images:
        instances = inference_detector(model, str(image), test_pipeline=pipeline).pred_instances
        boxes = instances.bboxes.cpu().tolist()
        scores = instances.scores.cpu().tolist()
        predictions[image.stem] = [dict(box=box, score=score)
                                   for box, score in zip(boxes, scores) if score >= CONF_THR]
    return predictions


def run_rtdetr(args, run_dir: Path, images: list[Path]) -> dict:
    from ultralytics import RTDETR

    model = RTDETR(str(args.weights))
    model.train(data=str(IR / "data.yaml"), imgsz=SIZE, epochs=args.epochs, batch=args.batch,
                workers=0, seed=42, device=0, amp=False, deterministic=False, mosaic=0.0,
                project=str(run_dir.parent),
                name=run_dir.name, exist_ok=True)
    model = RTDETR(str(run_dir / "weights" / "best.pt"))
    results = model.predict(source=str(images[0].parent), imgsz=SIZE, device=0,
                            conf=CONF_THR, stream=True, verbose=False)
    return {Path(result.path).stem: [dict(box=row[:4], score=row[4])
                                     for row in result.boxes.data.tolist()] for result in results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("dino", "rtdetr"))
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--dino-config", type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", type=Path, help="DINO 已有检查点，仅运行测试集推理")
    args = parser.parse_args()
    if args.model == "dino" and not args.dino_config:
        parser.error("DINO 需要 --dino-config")
    if not args.weights.is_file() or not (IR / "data.yaml").is_file():
        parser.error("缺少预训练权重或准备好的 IR 数据集")
    images = sorted((IR / "test" / "images").glob("*.png"))
    if not images:
        parser.error("IR 测试集为空")
    run_dir = PROJECT_ROOT / "runs" / "train" / args.name
    if args.checkpoint and (args.model != "dino" or not args.checkpoint.is_file()):
        parser.error("--checkpoint 需要已有的 DINO 检查点")
    run_dir.mkdir(parents=True, exist_ok=bool(args.checkpoint))
    if args.model == "dino":
        checkpoint = args.checkpoint or train_dino(args, run_dir)
        predictions = predict_dino(args, run_dir, checkpoint, images)
    else:
        predictions = run_rtdetr(args, run_dir, images)
    if set(predictions) != {image.stem for image in images}:
        raise RuntimeError("推理结果未覆盖全部测试图像")
    destination = run_dir / f"{args.name}_test_preds.json"
    pending = destination.with_suffix(".tmp")
    pending.write_text(json.dumps(predictions), encoding="utf-8")
    pending.replace(destination)
    print(f"逐图预测已保存: {destination}")


if __name__ == "__main__":
    main()
