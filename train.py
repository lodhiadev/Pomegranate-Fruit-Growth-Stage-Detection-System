# train.py
from ultralytics import YOLO
from pathlib import Path
from utils.curriculum_split import create_curriculum_splits

DEVICE = "mps"        # Apple Silicon GPU
IMGSZ = 768
BATCH = 16
EPOCHS_PER_PHASE = [10, 10, 15]   # easy → medium → hard

DATA_ROOT = Path("data")
TRAIN_IMAGES = DATA_ROOT / "images/train"
CURRICULUM_DIR = DATA_ROOT

def main():
    # 1️⃣ Collect all training images (GAN + real)
    train_images = list(TRAIN_IMAGES.glob("*.jpg"))
    print(f"Total training images found: {len(train_images)}")

    # 2️⃣ Create curriculum splits
    splits = create_curriculum_splits(
        train_list=train_images,
        output_dir=CURRICULUM_DIR
    )

    # 3️⃣ Load YOLOv8s pretrained model
    model = YOLO("yolov8s.pt")

    # 4️⃣ Curriculum training loop
    for idx, phase in enumerate(["easy", "medium", "hard"]):
        print(f"\n🚀 Training phase: {phase.upper()}")

        model.train(
            data="data/data.yaml",
            imgsz=IMGSZ,
            batch=BATCH,
            epochs=EPOCHS_PER_PHASE[idx],
            device=DEVICE,
            lr0=1e-3 if phase == "easy" else 5e-4,
            mosaic=0.6 if phase == "easy" else 0.2,
            patience=5,
            name="final_768",
            exist_ok=True
        )

    # 5️⃣ Save final weights explicitly
    final_weights = Path("runs/detect/final_768/weights/best.pt")
    print(f"\n✅ Final model saved at: {final_weights}")

if __name__ == "__main__":
    main()
