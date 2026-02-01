from ultralytics import YOLO

# ========================
# CONFIGURATION
# ========================
IMGSZ = 768
BATCH = 16
EPOCHS = 50
DEVICE = 0  # Kaggle GPU

# ========================
# LOAD MODEL
# ========================
model = YOLO("yolov8s.pt")

# ========================
# TRAIN
# ========================
model.train(
    data="/kaggle/input/pomegranate-dataset/pomegranate_dataset/yolo/data.yaml",
    imgsz=IMGSZ,
    batch=BATCH,
    epochs=EPOCHS,
    device=DEVICE,
    workers=8,
    optimizer="AdamW",
    lr0=5e-4,
    weight_decay=5e-4,
    warmup_epochs=3,
    mosaic=0.5,
    patience=10,
    name="yolov8s_gan_curriculum",
    save=True
)

print("✅ Training completed successfully")
