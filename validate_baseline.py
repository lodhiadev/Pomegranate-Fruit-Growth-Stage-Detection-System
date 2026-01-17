from ultralytics import YOLO

DEVICE = "mps"      # Apple GPU
IMGSZ = 768

# Load best trained baseline model
model = YOLO("/opt/homebrew/runs/detect/final_768/weights/best.pt")

# Run validation on val set
model.val(
    data="data/data.yaml",
    imgsz=IMGSZ,
    device=DEVICE
)
