from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
from prometheus_fastapi_instrumentator import Instrumentator
import cv2
import numpy as np
import io
import sqlite3
import os
from datetime import datetime
from collections import Counter

app = FastAPI(title="YOLOv8 Object Detection Service")

# 🚀 Initialize Prometheus Metrics
Instrumentator().instrument(app).expose(app)

# Load Model
model = YOLO('yolov8n.pt')

# --- DATABASE CONFIGURATION ---
DB_NAME = "data/detections.db"
MAX_LOG_ENTRIES = 1000  # Keep only the last 1000 detections to prevent "garbage"

def init_db():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(DB_NAME), exist_ok=True)
    
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                filename TEXT,
                object_class TEXT,
                confidence REAL
            )
        ''')
        conn.commit()

def cleanup_old_logs():
    """
    Optimization: Deletes old records if the table exceeds MAX_LOG_ENTRIES.
    This prevents the database from growing indefinitely.
    """
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        # Keep only the latest N rows
        cursor.execute(f'''
            DELETE FROM logs 
            WHERE id NOT IN (
                SELECT id FROM logs ORDER BY id DESC LIMIT {MAX_LOG_ENTRIES}
            )
        ''')
        conn.commit()

def log_detection(filename, results):
    """Logs detected objects into the database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        has_detections = False
        
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            obj_name = model.names[cls_id]
            
            # Log specific detection
            cursor.execute(
                "INSERT INTO logs (timestamp, filename, object_class, confidence) VALUES (?, ?, ?, ?)",
                (timestamp, filename, obj_name, conf)
            )
            has_detections = True
        
        conn.commit()
    
    # Trigger cleanup if we added new data
    if has_detections:
        cleanup_old_logs()

# Initialize DB on startup
init_db()
# --- END DATABASE CONFIGURATION ---

def process_image(contents):
    """Helper: Converts raw bytes to an OpenCV image."""
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.get("/")
def root():
    return {
        "message": "YOLOv8 Service is Running!",
        "docs": "/docs",
        "monitoring": "/metrics",
        "endpoints": {
            "/detect_image": "Returns image with bounding boxes",
            "/detect_json": "Returns JSON with object counts",
            "/history": "View detection logs",
            "/history/clear": "Clear all logs (DELETE)"
        }
    }

@app.post("/detect_image")
async def get_image_with_boxes(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold (0.0 - 1.0)")
):
    """
    Performs object detection and returns the annotated image.
    Logs detections to the database.
    """
    contents = await file.read()
    img = process_image(contents)

    results = model(img, conf=conf)
    
    # Log to DB
    log_detection(file.filename, results)

    plotted_img = results[0].plot()
    res, im_jpg = cv2.imencode(".jpg", plotted_img)
    
    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpeg")

@app.post("/detect_json")
async def get_object_counts(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0)
):
    """
    Performs object detection and returns a JSON summary.
    Logs detections to the database.
    """
    contents = await file.read()
    img = process_image(contents)

    results = model(img, conf=conf)
    
    # Log to DB
    log_detection(file.filename, results)
    
    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    counts = dict(Counter(detected_classes))

    return JSONResponse(content={
        "filename": file.filename,
        "total_objects": len(detected_classes),
        "breakdown": counts
    })

@app.get("/history")
def get_history(limit: int = 50):
    """Returns the latest detection logs."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            conn.row_factory = sqlite3.Row  # Return dict-like objects
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,))
            rows = [dict(row) for row in cursor.fetchall()]
        return {"latest_detections": rows}
    except Exception as e:
        return {"error": str(e)}

@app.delete("/history/clear")
def clear_history():
    """Clears the entire database log."""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM logs")
            conn.commit()
        return {"message": "All detection logs have been cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)