import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

from src.data_loader import load_data
from src.preprocess import clean_data, split_data, fit_scaler
from src.train_models import (
    train_random_forest,
    train_svm_grid,
    train_decision_tree,
    train_linear_regression,
    train_knn_grid,
)
from src.evaluate import evaluate_model
from src.model_utils import save_model, save_scaler
from src.compare_models import summarize_results

# --------------------- FastAPI Setup ---------------------
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aqi-predictor-five.vercel.app",  # frontend deployed URL
        "http://localhost:3000",                  # local dev frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend running!"}

# --------------------- Logging ---------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aqi-train")

# --------------------- Paths ---------------------
DATA_PATH = os.path.join("data", "aqi_dataset.csv")
MODEL_DIR = os.path.join("models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------- Training ---------------------
def main():
    logger.info("Loading data from %s", DATA_PATH)
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)
    results = {}

    # Random Forest
    logger.info("Training Random Forest")
    rf = train_random_forest(X_train, y_train)
    res_rf = evaluate_model(rf, X_test, y_test)
    results["RandomForest"] = res_rf

    X_train_scaled, X_test_scaled, scaler = fit_scaler(X_train, X_test)

    # SVM
    logger.info("Training SVM (GridSearch)")
    svr = train_svm_grid(X_train_scaled, y_train)
    res_svr = evaluate_model(svr, X_test_scaled, y_test)
    results["SVM"] = res_svr

    # Decision Tree
    dt = train_decision_tree(X_train_scaled, y_train)
    res_dt = evaluate_model(dt, X_test_scaled, y_test)
    results["DecisionTree"] = res_dt

    # Linear Regression
    lr = train_linear_regression(X_train_scaled, y_train)
    res_lr = evaluate_model(lr, X_test_scaled, y_test)
    results["LinearRegression"] = res_lr

    # KNN
    knn = train_knn_grid(X_train_scaled, y_train)
    res_knn = evaluate_model(knn, X_test_scaled, y_test)
    results["KNN"] = res_knn

    # Summarize results
    df_summary = summarize_results(results)
    best_row = df_summary.iloc[0]
    best_name = best_row["Model"]

    model_map = {
        "RandomForest": rf,
        "SVM": svr,
        "DecisionTree": dt,
        "LinearRegression": lr,
        "KNN": knn,
    }
    best_model = model_map[best_name]

    save_model(best_model, MODEL_PATH)
    if best_name != "RandomForest":
        save_scaler(scaler, SCALER_PATH)
    else:
        if os.path.exists(SCALER_PATH):
            os.remove(SCALER_PATH)

# --------------------- Prediction Endpoint ---------------------
class AQIData(BaseModel):
    PM2_5: float
    NO2: float
    SO2: float
    OZONE: float

# Load model and scaler at startup
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

@app.post("/predict")
def predict(data: AQIData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    features = np.array([[data.PM2_5, data.NO2, data.SO2, data.OZONE]])

    # Scale if scaler exists
    if scaler:
        features = scaler.transform(features)

    prediction = model.predict(features)[0]
    return {"AQI_prediction": float(prediction)}

# --------------------- Run Training ---------------------
if __name__ == "__main__":
    main()
