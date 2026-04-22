from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Loading model...")

# Load model
model = YOLO("/opt/homebrew/runs/detect/refine_896/weights/best.pt")

print("Running validation...")

# Run validation
metrics = model.val(
    data="data/data.yaml",
    imgsz=896,
    device="mps",
    plots=False
)

print("Validation complete")

# Get confusion matrix
cm = metrics.confusion_matrix.matrix

class_names = ["Bud", "Flower", "Early", "Mid", "Mature"]

print("Generating confusion matrix plot...")

plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt=".0f",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Stage")
plt.ylabel("Actual Stage")
plt.title("Confusion Matrix for Pomegranate Growth Stage Detection")

# Save file
output_path = "confusion_matrix_clean.png"
plt.tight_layout()
plt.savefig(output_path)

print(f"Saved confusion matrix to: {os.path.abspath(output_path)}")

plt.show()