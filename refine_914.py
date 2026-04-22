from ultralytics import YOLO

MODEL_PATH = "/opt/homebrew/runs/detect/best_914_model/weights/best.pt"
DATA_PATH = "/Users/devlodhia/Desktop/pomegranate_yolov8/data/data.yaml"

model = YOLO(MODEL_PATH)

model.train(
    data=DATA_PATH,
    imgsz=768,              # keep same resolution
    batch=16,
    epochs=20,              # short controlled run
    device="mps",

    # 🔒 Ultra-gentle refinement
    optimizer="AdamW",
    lr0=5e-5,               # very low LR
    weight_decay=0.0005,
    cos_lr=True,

    # 🚫 No augmentation in refinement phase
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,

    patience=8,
    amp=False,              # stable on MPS
    name="refine_from_914"
)