from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from ultralytics import YOLO
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic_settings import BaseSettings
from cap_from_youtube import cap_from_youtube
import cv2
import numpy as np
import io
import sqlite3
import os
import time
from datetime import datetime
from collections import Counter
from functools import lru_cache

# --- 1. CONFIGURATION ---
class Settings(BaseSettings):
    app_name: str = "YOLOv8 MLOps Service"
    model_path: str = "yolov8n.pt"
    db_path: str = "data/detections.db"
    max_log_entries: int = 1000
    video_source: str = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
    # Link to your code (Great for Portfolio)
    github_url: str = "https://github.com/Western-1" 

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
app = FastAPI(title=settings.app_name)

# --- 2. OBSERVABILITY ---
Instrumentator().instrument(app).expose(app)
model = YOLO(settings.model_path)

# --- 3. DATABASE LAYER ---
def init_db():
    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    with sqlite3.connect(settings.db_path) as conn:
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
    with sqlite3.connect(settings.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f'''
            DELETE FROM logs 
            WHERE id NOT IN (
                SELECT id FROM logs ORDER BY id DESC LIMIT {settings.max_log_entries}
            )
        ''')
        conn.commit()

def log_detection(filename, results):
    """Only logs uploaded images to DB, NOT video stream."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(settings.db_path) as conn:
        cursor = conn.cursor()
        has_detections = False
        for box in results[0].boxes:
            cursor.execute(
                "INSERT INTO logs (timestamp, filename, object_class, confidence) VALUES (?, ?, ?, ?)",
                (timestamp, filename, model.names[int(box.cls)], float(box.conf))
            )
            has_detections = True
        conn.commit()
    if has_detections:
        cleanup_old_logs()

init_db()

# --- 4. IMAGE & VIDEO PROCESSING ---
def process_image(contents):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def generate_frames(source, conf_threshold):
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
    elif "youtube.com" in source or "youtu.be" in source:
        try:
            cap = cap_from_youtube(source, '720p')
        except:
            return
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened(): return

    # Limit FPS to save CPU
    FPS_LIMIT = 30
    frame_duration = 1.0 / FPS_LIMIT

    while cap.isOpened():
        start_time = time.time()

        success, frame = cap.read()
        if not success:
            if "youtube" not in source and not str(source).isdigit():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else: break

        results = model(frame, conf=conf_threshold)
        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        
        elapsed = time.time() - start_time
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    cap.release()

# --- 5. UI & ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Dark Mode Dashboard UI"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{settings.app_name}</title>
        <style>
            body {{ background-color: #121212; color: #ffffff; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 0; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }}
            header {{ width: 100%; background-color: #1e1e1e; padding: 20px 0; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.5); margin-bottom: 40px; }}
            h1 {{ margin: 0; color: #00d4ff; font-weight: 300; letter-spacing: 1px; }}
            .container {{ width: 90%; max-width: 1000px; text-align: center; }}
            .video-wrapper {{ position: relative; border: 2px solid #333; border-radius: 12px; overflow: hidden; box-shadow: 0 0 30px rgba(0, 212, 255, 0.1); background: #000; }}
            img {{ width: 100%; display: block; }}
            .status-bar {{ margin-top: 20px; display: flex; justify-content: space-between; align-items: center; background: #1e1e1e; padding: 15px 25px; border-radius: 8px; }}
            .status-item {{ display: flex; align-items: center; font-size: 0.9rem; color: #aaa; }}
            .live-dot {{ height: 10px; width: 10px; background-color: #ff4444; border-radius: 50%; margin-right: 10px; box-shadow: 0 0 10px #ff4444; animation: pulse 2s infinite; }}
            @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.4; }} 100% {{ opacity: 1; }} }}
            .controls {{ margin-top: 40px; display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }}
            .btn {{ text-decoration: none; color: #fff; background: #2a2a2a; padding: 12px 25px; border-radius: 6px; border: 1px solid #333; transition: all 0.2s; font-weight: 500; display: flex; align-items: center; gap: 8px; }}
            .btn:hover {{ background: #00d4ff; color: #000; border-color: #00d4ff; transform: translateY(-2px); }}
            .footer {{ margin-top: auto; padding: 20px; color: #444; font-size: 0.8rem; }}
        </style>
    </head>
    <body>
        <header>
            <h1>YOLOv8 <span style="color: #fff;">MLOps Service</span></h1>
        </header>
        
        <div class="container">
            <div class="video-wrapper">
                <img src="/video_feed" alt="Live Stream">
            </div>
            
            <div class="status-bar">
                <div class="status-item">
                    <span class="live-dot"></span> SYSTEM ONLINE
                </div>
                <div class="status-item">
                    SOURCE: {settings.video_source if 'http' in settings.video_source else 'Local File'}
                </div>
                <div class="status-item">
                    MODEL: {settings.model_path}
                </div>
            </div>

            <div class="controls">
                <a href="/docs" class="btn" target="_blank">📄 API Docs</a>
                
                <a href="{settings.github_url}" class="btn" target="_blank">
                    <svg height="20" width="20" viewBox="0 0 16 16" fill="white"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>
                    GitHub Repo
                </a>

                <a href="/metrics" class="btn" target="_blank">📊 Metrics</a>
            </div>
        </div>

        <div class="footer">
            Powered by FastAPI, OpenCV & YOLOv8
        </div>
    </body>
    </html>
    """

@app.get("/video_feed", include_in_schema=False)
async def video_feed(conf: float = 0.4):
    return StreamingResponse(
        generate_frames(settings.video_source, conf), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/detect_image")
async def get_image_with_boxes(file: UploadFile = File(...), conf: float = Query(0.25)):
    contents = await file.read()
    results = model(process_image(contents), conf=conf)
    log_detection(file.filename, results)
    res, im_jpg = cv2.imencode(".jpg", results[0].plot())
    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpeg")

@app.post("/detect_json")
async def get_object_counts(file: UploadFile = File(...), conf: float = Query(0.25)):
    results = model(process_image(await file.read()), conf=conf)
    log_detection(file.filename, results)
    detected = [model.names[int(box.cls)] for box in results[0].boxes]
    return JSONResponse(content={"filename": file.filename, "objects": len(detected), "breakdown": dict(Counter(detected))})

@app.get("/history")
def get_history(limit: int = 50):
    with sqlite3.connect(settings.db_path) as conn:
        conn.row_factory = sqlite3.Row
        return {"latest_detections": [dict(row) for row in conn.cursor().execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]}

@app.delete("/history/clear")
def clear_history():
    with sqlite3.connect(settings.db_path) as conn: conn.cursor().execute("DELETE FROM logs").connection.commit()
    return {"message": "Logs cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)