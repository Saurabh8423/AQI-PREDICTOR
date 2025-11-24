# src/model_utils.py
import joblib
from typing import Any, Optional
import os

def save_model(model: Any, path: str):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str) -> Any:
    return joblib.load(path)

def save_scaler(scaler: Any, path: str):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path: str) -> Any:
    return joblib.load(path)
