import os
os.environ['TEMPORARILY_DISABLE_PROTOBUF_VERSION_CHECK'] = 'true'

import io
import urllib.request
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from contextlib import asynccontextmanager

# --- Global Model Artifact ---
model = None

# --- Lifespan Handler for Startup/Shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        # Check standard and absolute paths
        model_path = "stress_detection_model.h5"
        if not os.path.exists(model_path):
            model_path = "/home/xeeshan/4) FPY_paid_projects_docs/Projects/human_stress_detection/stress_detection_model.h5"
        
        if os.path.exists(model_path):
            print(f"[STATUS] Loading transfer-learned MobileNetV2 model from: {model_path}...")
            model = load_model(model_path)
            print("[SUCCESS] Stress Detection model loaded successfully and ready for inference.")
        else:
            print(f"[WARNING] Model file not found at {model_path}. Server will start, but predictions will load dynamically.")
    except Exception as e:
        print(f"[ERROR] Error during model startup: {str(e)}")
    yield

# --- FastAPI Initialization ---
app = FastAPI(
    title="CalmSphere AI — Human Stress Detection API",
    description="High-level FYP backend for AI-powered human stress detection via facial expression analysis.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schema for URL Predictions ---
class UrlPredictionRequest(BaseModel):
    url: str

# --- Preprocessing & Prediction Helper ---
def process_and_predict(img_bytes: bytes) -> dict:
    global model
    if model is None:
        # Attempt inline lazy load
        model_path = "stress_detection_model.h5"
        if not os.path.exists(model_path):
            model_path = "/home/xeeshan/4) FPY_paid_projects_docs/Projects/human_stress_detection/stress_detection_model.h5"
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            raise FileNotFoundError("stress_detection_model.h5 could not be located on the server.")

    # Open PIL Image and convert to RGB (strips Alpha channel from PNGs)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    # Preprocessing to match Keras training pipeline:
    # 1. Resize to 224x224
    # 2. Convert to array and normalize pixels (1/255.0)
    # 3. Expand dimensions to (1, 224, 224, 3)
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Inference
    prediction_raw = model.predict(img_array)[0][0]
    stress_prob = float(prediction_raw)
    non_stress_prob = float(1.0 - stress_prob)
    
    # Labels & Decision Rule (Keras alphabetical sorting: 'Non Stress' -> 0, 'Stress' -> 1)
    if stress_prob >= 0.50:
        label = "Stress"
        confidence = stress_prob
    else:
        label = "Non Stress"
        confidence = non_stress_prob

    # Generate personalized recommendations
    if label == "Stress":
        if confidence > 0.85:
            recommendation = "High stress detected. We strongly suggest taking a 5-minute deep breathing break. Try our breath pacer below, sip cold water, and step away from active screens."
            level = "Severe"
        else:
            recommendation = "Moderate stress detected. Your physiological markers suggest tension. Consider taking a walk, listening to our relaxing sounds hub, and stretching."
            level = "Moderate"
    else:
        recommendation = "Great! You look relaxed and composed. Continue maintaining this calm state. Keep staying hydrated and remember to take brief breaks throughout your day."
        level = "Normal"
        
    return {
        "status": "success",
        "label": label,
        "level": level,
        "confidence": confidence,
        "stress_probability": stress_prob,
        "non_stress_probability": non_stress_prob,
        "recommendation": recommendation
    }

# --- API Endpoints ---

@app.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = process_and_predict(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_url")
async def predict_url(request: UrlPredictionRequest):
    try:
        # Download image from URL safely using built-in urllib
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        req = urllib.request.Request(request.url, headers=headers)
        with urllib.request.urlopen(req) as response:
            img_bytes = response.read()
            
        result = process_and_predict(img_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not retrieve or analyze image from the provided URL: {str(e)}")

@app.get("/")
def serve_frontend():
    index_path = "index.html"
    if not os.path.exists(index_path):
        index_path = "/home/xeeshan/4) FPY_paid_projects_docs/Projects/human_stress_detection/index.html"
    return FileResponse(index_path)

if __name__ == "__main__":
    uvicorn.run("backend_app:app", host="127.0.0.1", port=8000, reload=False)
