import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

ROOT = os.path.dirname(os.path.dirname(__file__))  # backend/app -> backend
MODEL_PATH = os.path.join(ROOT, "models", "best_model.pkl")
SCALER_PATH = os.path.join(ROOT, "models", "scaler.pkl")

app = FastAPI(title="AQI Predictor API", version="1.0")

# 🔥 Fix CORS issue here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AQIRequest(BaseModel):
    PM2_5: float
    NO2: float
    SO2: float
    OZONE: float

_model = None
_scaler = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found at {MODEL_PATH}. Run training first.")
        _model = joblib.load(MODEL_PATH)
    return _model

def get_scaler():
    global _scaler
    if _scaler is None:
        if os.path.exists(SCALER_PATH):
            _scaler = joblib.load(SCALER_PATH)
    return _scaler

@app.on_event("startup")
def startup_event():
    try:
        get_model()
        get_scaler()
    except Exception as e:
        print("Warning during startup:", e)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: AQIRequest):
    features = np.array([[payload.PM2_5, payload.NO2, payload.SO2, payload.OZONE]], dtype=float)

    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    scaler = get_scaler()
    if scaler is not None:
        features = scaler.transform(features)

    try:
        pred = model.predict(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {str(e)}")

    return {"AQI_prediction": float(pred[0])}
