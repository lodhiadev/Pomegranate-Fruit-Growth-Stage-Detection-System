import io
import qrcode
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO
import uvicorn

# -------- CONFIG --------
MODEL_PATH = "/opt/homebrew/runs/detect/refine_896/weights/best.pt"
CONF_THRES = 0.30
# ------------------------

CLASS_NAMES = {
    0: "Bud",
    1: "Flower",
    2: "Early",
    3: "Mid",
    4: "Mature"
}

model = YOLO(MODEL_PATH)

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model.predict(source=image, conf=CONF_THRES, verbose=False)

    r = results[0]

    detections = []

    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)

            detections.append({
                "stage_id": cls_id,
                "stage_name": CLASS_NAMES[cls_id],
                "confidence": round(conf*100,2)
            })

    harvest_ready = any(d["stage_id"] == 4 for d in detections)

    return {
        "detections": detections,
        "harvest_ready": harvest_ready
    }


if __name__ == "__main__":

    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    url = f"http://{local_ip}:8000"

    print("\nServer running at:", url)

    qr = qrcode.make(url)
    qr.show()

    uvicorn.run(app, host="0.0.0.0", port=8000)