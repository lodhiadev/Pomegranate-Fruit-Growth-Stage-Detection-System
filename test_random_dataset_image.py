import os
import random
import cv2
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "/opt/homebrew/runs/detect/curriculum_phase3/weights/best.pt"
IMAGE_DIR = "data/images/val"   # change to train if needed
CONF_THRES = 0.20
OUTPUT_IMAGE = "random_test_result.jpg"
# ---------------------------------------

CLASS_NAMES = {
    0: "bud",
    1: "flower",
    2: "early",
    3: "mid",
    4: "mature"
}

# Pick random image
images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
assert len(images) > 0, "No images found in dataset folder!"

img_name = random.choice(images)
img_path = os.path.join(IMAGE_DIR, img_name)

print(f"\n📸 Random test image selected: {img_path}")

# Load model
model = YOLO(MODEL_PATH)

# Run inference
results = model(img_path, conf=CONF_THRES)

# Draw detections
r = results[0]
img = r.orig_img.copy()

if r.boxes is not None:
    for box in r.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# Save result
cv2.imwrite(OUTPUT_IMAGE, img)

print(f"✅ Result saved as: {OUTPUT_IMAGE}")
