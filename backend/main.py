# main.py
"""
Run this script to train models, compare them, and save the best model (and scaler if used).
"""

import os
import logging
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
from fastapi import FastAPI

app = FastAPI()
@app.get("/")
def home():
    return {"message": "Backend running!"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("aqi-train")

DATA_PATH = os.path.join("data", "aqi_dataset.csv")
MODEL_DIR = os.path.join("models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    logger.info("Loading data from %s", DATA_PATH)
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    results = {}

    # Random Forest (works with raw features)
    logger.info("Training Random Forest")
    rf = train_random_forest(X_train, y_train)
    res_rf = evaluate_model(rf, X_test, y_test)
    results["RandomForest"] = res_rf
    logger.info("RandomForest - MAE: %.4f, R2: %.4f", res_rf["mae"], res_rf["r2"])

    # For models that prefer scaled features, fit scaler
    logger.info("Fitting scaler for models that require scaling")
    X_train_scaled, X_test_scaled, scaler = fit_scaler(X_train, X_test)

    # SVM (grid search)
    logger.info("Training SVM (GridSearch)")
    svr = train_svm_grid(X_train_scaled, y_train)
    res_svr = evaluate_model(svr, X_test_scaled, y_test)
    results["SVM"] = res_svr
    logger.info("SVM - MAE: %.4f, R2: %.4f", res_svr["mae"], res_svr["r2"])

    # Decision Tree
    logger.info("Training Decision Tree")
    dt = train_decision_tree(X_train_scaled, y_train)
    res_dt = evaluate_model(dt, X_test_scaled, y_test)
    results["DecisionTree"] = res_dt
    logger.info("DecisionTree - MAE: %.4f, R2: %.4f", res_dt["mae"], res_dt["r2"])

    # Linear Regression
    logger.info("Training Linear Regression")
    lr = train_linear_regression(X_train_scaled, y_train)
    res_lr = evaluate_model(lr, X_test_scaled, y_test)
    results["LinearRegression"] = res_lr
    logger.info("LinearRegression - MAE: %.4f, R2: %.4f", res_lr["mae"], res_lr["r2"])

    # KNN (grid)
    logger.info("Training KNN (GridSearch)")
    knn = train_knn_grid(X_train_scaled, y_train)
    res_knn = evaluate_model(knn, X_test_scaled, y_test)
    results["KNN"] = res_knn
    logger.info("KNN - MAE: %.4f, R2: %.4f", res_knn["mae"], res_knn["r2"])

    # Summarize results
    df_summary = summarize_results(results)
    logger.info("\nModel comparison:\n%s", df_summary.to_string(index=False))

    # Choose best by R2
    best_row = df_summary.iloc[0]
    best_name = best_row["Model"]
    logger.info("Best model by R2: %s (R2=%.4f)", best_name, float(best_row["R2"]))

    # Map name to object
    model_map = {
        "RandomForest": rf,
        "SVM": svr,
        "DecisionTree": dt,
        "LinearRegression": lr,
        "KNN": knn,
    }

    best_model = model_map[best_name]

    # Save model
    save_model(best_model, MODEL_PATH)
    logger.info("Saved best model at %s", MODEL_PATH)

    # Save scaler if best model requires scaled input (SVM, LR, KNN, DT were trained on scaled data)
    # We'll conservatively save the scaler if best model is not RandomForest.
    if best_name != "RandomForest":
        save_scaler(scaler, SCALER_PATH)
        logger.info("Saved scaler at %s", SCALER_PATH)
    else:
        # Remove scaler file if exists (to avoid confusion)
        if os.path.exists(SCALER_PATH):
            os.remove(SCALER_PATH)
            logger.info("Removed existing scaler (not needed for RandomForest)")

if __name__ == "__main__":
    main()
