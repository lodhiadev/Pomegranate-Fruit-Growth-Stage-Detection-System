import os
import random
import cv2
from collections import Counter
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "/opt/homebrew/runs/detect/curriculum_phase3/weights/best.pt"
IMAGE_DIR = "data/images/val"      # or "data/images/train"
NUM_IMAGES = 10                    # how many random images to test
CONF_THRES = 0.25
OUTPUT_DIR = "random_batch_results"
# ---------------------------------------

CLASS_NAMES = {
    0: "bud",
    1: "flower",
    2: "early",
    3: "mid",
    4: "mature"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Collect images
images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".jpg")]
assert len(images) >= NUM_IMAGES, "Not enough images in the folder!"

# Pick random images
sampled_images = random.sample(images, NUM_IMAGES)

print(f"\n🔍 Testing {NUM_IMAGES} random images...\n")

for i, img_name in enumerate(sampled_images, start=1):
    img_path = os.path.join(IMAGE_DIR, img_name)
    results = model(img_path, conf=CONF_THRES)

    r = results[0]
    img = r.orig_img.copy()
    counts = Counter()

    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            counts[cls_id] += 1

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

    out_path = os.path.join(OUTPUT_DIR, f"result_{i}_{img_name}")
    cv2.imwrite(out_path, img)

    print(f"Image {i}: {img_name}")
    for cid, cname in CLASS_NAMES.items():
        print(f"  {cname}: {counts[cid]}")
    print("-" * 40)

print(f"\n✅ Done. Annotated images saved in '{OUTPUT_DIR}/'")
