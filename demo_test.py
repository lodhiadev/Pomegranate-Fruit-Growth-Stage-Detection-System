import os
import random
import time
from collections import Counter
from ultralytics import YOLO

# -------- CONFIG --------
MODEL_PATH = "/opt/homebrew/runs/detect/refine_896/weights/best.pt"
IMAGE_DIR = "data/images/val"
CONF_THRES = 0.30
DEVICE = "mps"
HARVEST_CONF_THRESHOLD = 0.80  # mature confidence threshold
# ------------------------

# Stage legend
CLASS_NAMES = {
    0: "Bud",
    1: "Flower",
    2: "Early",
    3: "Mid",
    4: "Mature"
}

# Select random image
images = [f for f in os.listdir(IMAGE_DIR)
          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

assert len(images) > 0, "No images found in directory."

img_name = random.choice(images)
IMAGE_PATH = os.path.join(IMAGE_DIR, img_name)

# Load model
model = YOLO(MODEL_PATH)

start = time.time()

results = model.predict(
    source=IMAGE_PATH,
    conf=CONF_THRES,
    device=DEVICE,
    verbose=False
)

end = time.time()

r = results[0]

print("\n====== POMEGRANATE MATURITY ANALYSIS ======")
print(f"Selected Image: {IMAGE_PATH}")

stage_counts = []
mature_confidences = []

if r.boxes is not None and len(r.boxes) > 0:
    print("\nDetected Stages:")
    for box in r.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        stage = CLASS_NAMES.get(cls_id, "Unknown")

        stage_counts.append(cls_id)

        if cls_id == 4:
            mature_confidences.append(conf)

        print(f"  → Stage {cls_id} ({stage}) : {conf*100:.2f}%")

    # Majority stage
    majority_stage = Counter(stage_counts).most_common(1)[0][0]

    # Harvest decision logic
    if 4 in stage_counts:
        max_mature_conf = max(mature_confidences)
        if max_mature_conf >= HARVEST_CONF_THRESHOLD:
            decision = "✅ HARVEST READY"
        else:
            decision = "⚠ Mature detected but low confidence"
    else:
        decision = "❌ NOT READY"

else:
    decision = "No fruit detected."

print("\nLegend:")
for k, v in CLASS_NAMES.items():
    print(f"  {k} = {v}")

print("\nHarvest Decision:")
print(f"  → {decision}")

print("\nPerformance:")
print(f"  → Inference Time: {(end - start)*1000:.2f} ms")

print("===========================================\n")