import os
import random
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "/opt/homebrew/runs/detect/refine_896/weights/best.pt"
IMAGE_DIR = "data/images/val"
NUM_IMAGES = 10
CONF_THRES = 0.30
# ----------------------------------------

# Load model
model = YOLO(MODEL_PATH)

# Collect images
images = [f for f in os.listdir(IMAGE_DIR)
          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

assert len(images) >= NUM_IMAGES, "Not enough images in folder!"

# Randomly select images
selected = random.sample(images, NUM_IMAGES)

print(f"\nRunning inference on {NUM_IMAGES} random images...\n")

for idx, img_name in enumerate(selected, 1):
    img_path = os.path.join(IMAGE_DIR, img_name)

    print(f"Image {idx}: {img_path}")

    results = model.predict(
        source=img_path,
        conf=CONF_THRES,
        device="mps",
        save=True
    )

    r = results[0]

    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            print(f"   → Class {cls_id}, Confidence: {conf:.2f}")
    else:
        print("   → No detections")

    print("-" * 40)

print("\n✅ All results saved to runs/detect/predict/")