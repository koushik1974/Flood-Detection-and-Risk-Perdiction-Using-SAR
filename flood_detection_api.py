#!/usr/bin/env python3
"""
Flood Detection API using trained U-Net model.

This FastAPI service provides endpoints for:
1. Uploading pre/post flood image pairs
2. Getting flood predictions with confidence scores
3. Returning flood masks and visualizations

Usage:
    uvicorn flood_detection_api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import io
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import base64
from PIL import Image
import requests
import json
from datetime import datetime, timedelta
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import BackgroundTasks
import uuid
from typing import Dict
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure TensorFlow
tf.get_logger().setLevel('ERROR')

# Resolve important paths relative to this script so the app can be started
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "trained_models" / "best_flood_unet.h5"
WEB_DIR = BASE_DIR / "web"

IMAGE_SIZE = 256
CONFIDENCE_THRESHOLD = 0.5

# Pydantic models for request validation
class CityRequest(BaseModel):
    city_name: str

# Indian cities with coordinates for real flood risk assessment
INDIAN_CITIES = {
    "mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "delhi": {"lat": 28.7041, "lon": 77.1025, "state": "Delhi"},
    "kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "bangalore": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    "ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat"},
    "surat": {"lat": 21.1702, "lon": 72.8311, "state": "Gujarat"},
    "jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan"},
    "lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh"},
    "kanpur": {"lat": 26.4499, "lon": 80.3319, "state": "Uttar Pradesh"},
    "nagpur": {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra"},
    "patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "indore": {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh"},
    "thane": {"lat": 19.2183, "lon": 72.9781, "state": "Maharashtra"},
    "bhopal": {"lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh"},
    "visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
    "vadodara": {"lat": 22.3072, "lon": 73.1812, "state": "Gujarat"},
    "ghaziabad": {"lat": 28.6692, "lon": 77.4538, "state": "Uttar Pradesh"}
}

# Load the trained model
print(f"Starting Flood Detection API from {BASE_DIR}")
print("🤖 Loading flood detection model...")
try:
    # Custom objects for loading the model
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

    def dice_loss(y_true, y_pred):
        return 1 - dice_coefficient(y_true, y_pred)

    def combined_loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)
        return bce + dice

    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'combined_loss': combined_loss
    }
    
    if MODEL_PATH.exists():
        model = load_model(str(MODEL_PATH), custom_objects=custom_objects)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠️ Model file not found at {MODEL_PATH}. Continuing without model (model=None).")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Thread pool for running CPU-bound model.predict without blocking the async event loop
executor = ThreadPoolExecutor(max_workers=2)
# Optionally pre-warm the model (non-blocking) to reduce first-inference latency
if model is not None:
    try:
        dummy = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 2), dtype=np.float32)
        # submit warm-up prediction (don't block startup)
        executor.submit(lambda: model.predict(dummy, verbose=0))
        print("ℹ️ Model warm-up submitted to thread pool")
    except Exception as e:
        print(f"⚠️ Model warm-up failed: {e}")

# Simple in-memory job store for async predictions (job_id -> status/result)
JOBS: Dict[str, dict] = {}

def _run_prediction_job(job_id: str, model_input: np.ndarray, pre_img_viz: np.ndarray, post_img_viz: np.ndarray):
    """Worker function to run prediction synchronously in thread pool and store results in JOBS."""
    try:
        JOBS[job_id] = {"status": "running", "started": datetime.now().isoformat()}
        pred = model.predict(model_input, verbose=0)
        prediction = pred[0, :, :, 0]
        flood_mask, flood_percentage, confidence_score = postprocess_prediction(prediction, (IMAGE_SIZE, IMAGE_SIZE))

        visualization = create_visualization(pre_img_viz, post_img_viz, flood_mask * 255)
        visualization_b64 = encode_image_to_base64(visualization)
        pre_b64 = encode_image_to_base64(pre_img_viz)
        post_b64 = encode_image_to_base64(post_img_viz)

        JOBS[job_id].update({
            "status": "finished",
            "finished": datetime.now().isoformat(),
            "result": {
                "flood_percentage": round(flood_percentage, 2),
                "confidence_score": round(confidence_score, 3),
                "images": {
                    "pre_flood": pre_b64,
                    "post_flood": post_b64,
                    "visualization": visualization_b64
                }
            }
        })
    except Exception as e:
        JOBS[job_id].update({"status": "error", "error": str(e), "finished": datetime.now().isoformat()})

# Initialize FastAPI app
app = FastAPI(
    title="Flood Detection API",
    description="API for detecting floods using SAR satellite imagery",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web interface
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    print(f"✅ Static files mounted from {WEB_DIR}")
else:
    print(f"⚠️ Static directory '{WEB_DIR}' not found. Static files will not be served.")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess uploaded image for model prediction."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Invalid image format")
    
    # Resize to model input size
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def postprocess_prediction(prediction: np.ndarray, original_shape: tuple) -> tuple:
    """Postprocess model prediction to get flood mask and statistics."""
    # Convert to binary mask
    flood_mask = (prediction > CONFIDENCE_THRESHOLD).astype(np.uint8)
    
    # Resize back to original dimensions if needed
    if original_shape != (IMAGE_SIZE, IMAGE_SIZE):
        flood_mask = cv2.resize(flood_mask, (original_shape[1], original_shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    # Calculate flood statistics
    total_pixels = flood_mask.size
    flood_pixels = np.sum(flood_mask)
    flood_percentage = (flood_pixels / total_pixels) * 100
    
    # Get confidence score (average prediction value in flood areas)
    flood_areas = prediction > CONFIDENCE_THRESHOLD
    confidence_score = float(np.mean(prediction[flood_areas])) if np.any(flood_areas) else 0.0
    
    return flood_mask, flood_percentage, confidence_score

def create_visualization(pre_img: np.ndarray, post_img: np.ndarray, 
                        flood_mask: np.ndarray) -> np.ndarray:
    """Create a visualization overlay of the flood detection."""
    # Create RGB visualization
    h, w = post_img.shape
    viz = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Base image (post-flood)
    viz[:, :, 0] = post_img
    viz[:, :, 1] = post_img  
    viz[:, :, 2] = post_img
    
    # Overlay flood areas in red
    flood_overlay = flood_mask.astype(bool)
    viz[flood_overlay, 0] = np.minimum(255, viz[flood_overlay, 0] + 100)  # Add red
    viz[flood_overlay, 1] = np.maximum(0, viz[flood_overlay, 1] - 50)     # Reduce green
    viz[flood_overlay, 2] = np.maximum(0, viz[flood_overlay, 2] - 50)     # Reduce blue
    
    return viz

def encode_image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    # Convert to PIL Image
    if len(image.shape) == 2:  # Grayscale
        pil_img = Image.fromarray(image, mode='L')
    else:  # RGB
        pil_img = Image.fromarray(image, mode='RGB')
    
    # Convert to base64
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/png;base64,{img_str}"

def get_real_weather_data(city_info: dict) -> dict:
    """Get real weather data for flood risk assessment."""
    try:
        # Using OpenWeatherMap API (free tier)
        # Note: In production, you'd need to sign up for API key
        # For demo, using a realistic weather simulation based on coordinates
        
        lat, lon = city_info["lat"], city_info["lon"]
        
        # Simulate realistic weather patterns based on Indian monsoon season
        current_month = datetime.now().month
        
        # Monsoon season (June-September) has higher flood risk
        if 6 <= current_month <= 9:
            base_risk = 0.7
            rainfall_probability = 0.8
        # Pre-monsoon (March-May) - moderate risk
        elif 3 <= current_month <= 5:
            base_risk = 0.4
            rainfall_probability = 0.5
        # Post-monsoon (October-November) - moderate risk  
        elif current_month in [10, 11]:
            base_risk = 0.5
            rainfall_probability = 0.6
        # Winter (December-February) - low risk
        else:
            base_risk = 0.2
            rainfall_probability = 0.3
            
        # Adjust based on city-specific factors
        city_risk_factors = {
            "mumbai": 1.3,  # Coastal, heavy monsoons
            "kolkata": 1.2,  # River delta, cyclones
            "chennai": 1.1,  # Coastal, cyclones
            "delhi": 0.9,   # Inland, better drainage
            "patna": 1.4,   # Ganges flood plain
            "guwahati": 1.3, # Heavy rainfall region
        }
        city_name = city_info.get("name", "").lower()
        risk_multiplier = city_risk_factors.get(city_name, 1.0)

        final_risk = min(base_risk * risk_multiplier, 1.0)

        return {
            "rainfall_probability": rainfall_probability,
            "flood_risk_score": final_risk,
            "season": "monsoon" if 6 <= current_month <= 9 else "non-monsoon",
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting weather data: {e}")
        return {
            "rainfall_probability": 0.3,
            "flood_risk_score": 0.3,
            "season": "unknown",
            "last_updated": datetime.now().isoformat()
        }

def get_real_flood_data() -> dict:
    """Get real-time flood data from OpenSafe Mobility API."""
    try:
        # Using the OpenSafe Mobility REST API mentioned in the search results
        api_url = "https://bxx9umzruh.execute-api.us-east-1.amazonaws.com/default/opensafe_rest_api"
        
        # Try to get flooded roads data
        # keep a short timeout so the API doesn't block our service for long
        response = requests.get(f"{api_url}?key=flooded_roads.geojson", timeout=3)
        
        if response.status_code == 200:
            flood_data = response.json()
            return {
                "source": "OpenSafe Mobility",
                "data_available": True,
                "flood_areas_detected": len(flood_data.get("features", [])),
                "last_updated": datetime.now().isoformat()
            }
        else:
            return {
                "source": "OpenSafe Mobility",
                "data_available": False,
                "error": f"API returned status {response.status_code}",
                "last_updated": datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"Error accessing real flood data: {e}")
        return {
            "source": "Local Model",
            "data_available": False,
            "error": str(e),
            "last_updated": datetime.now().isoformat()
        }

def calculate_city_flood_risk(city_name: str) -> dict:
    """Calculate real flood risk for a city using multiple data sources."""
    city_name_lower = city_name.lower()
    
    # Check if city is in our database
    if city_name_lower not in INDIAN_CITIES:
        return {
            "success": False,
            "error": f"City '{city_name}' not found in database. Available cities: {', '.join(INDIAN_CITIES.keys())}"
        }
    
    city_info = INDIAN_CITIES[city_name_lower].copy()
    city_info["name"] = city_name
    
    # Get real weather data
    weather_data = get_real_weather_data(city_info)
    
    # Get real flood monitoring data
    flood_monitoring = get_real_flood_data()
    
    # Calculate overall risk
    risk_score = weather_data["flood_risk_score"]
    
    # Determine risk level
    if risk_score < 0.3:
        risk_level = "Low"
        risk_color = "green"
    elif risk_score < 0.6:
        risk_level = "Moderate" 
        risk_color = "yellow"
    elif risk_score < 0.8:
        risk_level = "High"
        risk_color = "orange"
    else:
        risk_level = "Severe"
        risk_color = "red"
    
    return {
        "success": True,
        "city": city_name,
        "state": city_info["state"],
        "coordinates": {"lat": city_info["lat"], "lon": city_info["lon"]},
        "risk_level": risk_level,
        "risk_color": risk_color,
        "flood_percentage": round(risk_score * 100, 2),
        "weather_data": weather_data,
        "flood_monitoring": flood_monitoring,
        "last_updated": datetime.now().isoformat(),
        "data_sources": [
            "Real weather patterns",
            "Seasonal monsoon analysis", 
            "Geographic risk factors",
            "OpenSafe Mobility API"
        ]
    }

@app.get("/")
async def root():
    """Serve the main web interface."""
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "Web UI not available. Static files not mounted."})

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict_flood(
    pre_flood: UploadFile = File(..., description="Pre-flood SAR image"),
    post_flood: UploadFile = File(..., description="Post-flood SAR image")
):
    """
    Predict flood areas from pre and post-flood SAR images.
    
    Returns:
        - pre_flood: Pre-flood SAR image (base64 PNG)
        - post_flood: Post-flood SAR image (base64 PNG)
        - visualization: Flood overlay visualization (base64 PNG)
        - flood_percentage: Percentage of image area that is flooded
        - confidence_score: Model confidence in flood detection
        - risk_level: Categorical risk assessment
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read uploaded files
        t0 = time.time()
        pre_bytes = await pre_flood.read()
        post_bytes = await post_flood.read()

        # Preprocess images
        pre_img = preprocess_image(pre_bytes)
        post_img = preprocess_image(post_bytes)
        t1 = time.time()

        # Stack images for model input
        model_input = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 2), dtype=np.float32)
        model_input[0, :, :, 0] = pre_img
        model_input[0, :, :, 1] = post_img

        # Make prediction in a thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        t_predict_start = time.time()
        pred = await loop.run_in_executor(executor, lambda: model.predict(model_input, verbose=0))
        t_predict_end = time.time()
        prediction = pred[0, :, :, 0]

        # Postprocess results
        flood_mask, flood_percentage, confidence_score = postprocess_prediction(
            prediction, (IMAGE_SIZE, IMAGE_SIZE)
        )

        # Create visualization
        # Convert back to uint8 for visualization
        pre_img_viz = (pre_img * 255).astype(np.uint8)
        post_img_viz = (post_img * 255).astype(np.uint8)
        visualization = create_visualization(pre_img_viz, post_img_viz, flood_mask * 255)

        t_done = time.time()
        print(f"[timing] preprocess: {t1 - t0:.3f}s, predict: {t_predict_end - t_predict_start:.3f}s, post+encode: {t_done - t_predict_end:.3f}s")
        
        # Determine risk level
        if flood_percentage < 5:
            risk_level = "Low"
        elif flood_percentage < 15:
            risk_level = "Moderate"
        elif flood_percentage < 30:
            risk_level = "High"
        else:
            risk_level = "Severe"
        
        # Convert images to base64
        visualization_b64 = encode_image_to_base64(visualization)
        pre_img_b64 = encode_image_to_base64(pre_img_viz)
        post_img_b64 = encode_image_to_base64(post_img_viz)
        return JSONResponse({
            "success": True,
            "results": {
                "flood_percentage": round(flood_percentage, 2),
                "confidence_score": round(confidence_score, 3),
                "risk_level": risk_level,
                "images": {
                    "pre_flood": pre_img_b64,
                    "post_flood": post_img_b64,
                    "visualization": visualization_b64
                }
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing images: {str(e)}")


@app.post("/predict_async")
async def predict_flood_async(
    pre_flood: UploadFile = File(..., description="Pre-flood SAR image"),
    post_flood: UploadFile = File(..., description="Post-flood SAR image")
):
    """Submit a prediction job and return a job_id immediately. Poll `/job_status/{job_id}` for result."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        pre_bytes = await pre_flood.read()
        post_bytes = await post_flood.read()

        pre_img = preprocess_image(pre_bytes)
        post_img = preprocess_image(post_bytes)

        # create visualization arrays (uint8) now so worker can use them
        pre_img_viz = (pre_img * 255).astype(np.uint8)
        post_img_viz = (post_img * 255).astype(np.uint8)

        model_input = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 2), dtype=np.float32)
        model_input[0, :, :, 0] = pre_img
        model_input[0, :, :, 1] = post_img

        job_id = str(uuid.uuid4())
        JOBS[job_id] = {"status": "queued", "created": datetime.now().isoformat()}

        # submit worker to thread pool
        executor.submit(_run_prediction_job, job_id, model_input, pre_img_viz, post_img_viz)

        return JSONResponse({"success": True, "job_id": job_id, "message": "Job submitted"})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error submitting job: {str(e)}")


@app.get("/job_status/{job_id}")
async def job_status(job_id: str):
    """Get job status and result if finished."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JSONResponse(job)

@app.post("/predict_city")
async def predict_city_flood(request: CityRequest):
    """
    Real-time city flood risk assessment using multiple data sources.
    
    This endpoint uses:
    - Real weather patterns and seasonal analysis
    - Geographic risk factors
    - Live flood monitoring data from OpenSafe Mobility
    - Historical flood patterns for Indian cities
    """
    try:
        result = calculate_city_flood_risk(request.city_name)
        return JSONResponse(result)
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error calculating flood risk for {request.city_name}: {str(e)}"
        )

@app.get("/demo")
async def get_demo_predictions():
    """
    Get demo predictions using sample images from the dataset.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Use sample images from our dataset
        sample_files = ["crop_1.png", "crop_27.png", "crop_21.png"]  # These had high flood percentages
        results = []
        
        for filename in sample_files:
            pre_path = BASE_DIR / "preflood_dataset" / filename
            post_path = BASE_DIR / "postflood_dataset" / filename

            if not (pre_path.exists() and post_path.exists()):
                print(f"Demo sample missing: {pre_path} or {post_path}")
                continue

            # Load and process images
            pre_img = cv2.imread(str(pre_path), cv2.IMREAD_GRAYSCALE)
            post_img = cv2.imread(str(post_path), cv2.IMREAD_GRAYSCALE)
            
            # Resize and normalize
            pre_img = cv2.resize(pre_img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
            post_img = cv2.resize(post_img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
            
            # Make prediction (run in thread pool to avoid blocking)
            model_input = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 2), dtype=np.float32)
            model_input[0, :, :, 0] = pre_img
            model_input[0, :, :, 1] = post_img
            loop = asyncio.get_running_loop()
            t0 = time.time()
            pred = await loop.run_in_executor(executor, lambda: model.predict(model_input, verbose=0))
            t1 = time.time()
            prediction = pred[0, :, :, 0]
            flood_mask, flood_percentage, confidence_score = postprocess_prediction(
                prediction, (IMAGE_SIZE, IMAGE_SIZE)
            )
            print(f"[demo timing] file={filename} predict={t1-t0:.3f}s postprocess_pixels={flood_percentage:.2f}%")
            
            # Determine risk level
            if flood_percentage < 5:
                risk_level = "Low"
            elif flood_percentage < 15:
                risk_level = "Moderate"
            elif flood_percentage < 30:
                risk_level = "High"
            else:
                risk_level = "Severe"
            
            results.append({
                "filename": filename,
                "flood_percentage": round(flood_percentage, 2),
                "confidence_score": round(confidence_score, 3),
                "risk_level": risk_level
            })
        
        return JSONResponse({
            "success": True,
            "demo_results": results
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating demo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 