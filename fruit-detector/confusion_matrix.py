from ultralytics import YOLO

# Load trained model
model = YOLO("/opt/homebrew/runs/detect/refine_896/weights/best.pt")

# Run validation on dataset
metrics = model.val(
    data="data.yaml",   # your dataset config
    imgsz=896,
    device="mps",
    plots=True
)

print("\nValidation finished.")
print(metrics)