import torch
from ultralytics import YOLO

torch.set_default_dtype(torch.float32)

DEVICE = "mps"
IMGSZ = 896
BATCH = 8   # Reduce batch — 896 needs more memory

model = YOLO("/opt/homebrew/runs/detect/best_927_model.pt")

model.train(
    data="/Users/devlodhia/Desktop/pomegranate_yolov8/data/data.yaml",
    imgsz=IMGSZ,
    batch=BATCH,
    epochs=15,
    device=DEVICE,
    optimizer="AdamW",
    lr0=8e-5,            # Lower than before (gentle refinement)
    weight_decay=0.0005,
    mosaic=0.0,
    cos_lr=True,
    patience=7,
    amp=False,
    name="refine_896"
)