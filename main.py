from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import io
from collections import Counter

app = FastAPI(title="YOLOv8 Object Detection Service")

# Завантажуємо модель
model = YOLO('yolov8n.pt')

@app.get("/")
def root():
    return {
        "message": "Welcome to Object Detection API!",
        "endpoints": {
            "/detect_image": "Returns image with bounding boxes",
            "/detect_json": "Returns JSON with object counts"
        }
    }

def process_image(contents):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.post("/detect_image")
async def get_image_with_boxes(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Minimum confidence threshold (0.0 - 1.0)")
):
    """
    conf standart = 0.25
    """
    contents = await file.read()
    img = process_image(contents)

    results = model(img, conf=conf)

    plotted_img = results[0].plot()

    res, im_jpg = cv2.imencode(".jpg", plotted_img)
    
    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpeg")

@app.post("/detect_json")
async def get_object_counts(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0)
):
    """
    Return json file with stats of detected persons or objects
    """
    contents = await file.read()
    img = process_image(contents)

    results = model(img, conf=conf)
    
    detected_classes = []
    for box in results[0].boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        detected_classes.append(class_name)

    counts = dict(Counter(detected_classes))

    return JSONResponse(content={
        "filename": file.filename,
        "total_objects": len(detected_classes),
        "breakdown": counts
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)