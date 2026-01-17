from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train2/weights/last.pt")

    model.train(
        data="data/data.yaml",
        epochs=100,        # total target epochs
        resume=True,       # VERY IMPORTANT
        imgsz=640,
        batch=16,
        device="mps"       # Apple GPU
    )

if __name__ == "__main__":
    main()
