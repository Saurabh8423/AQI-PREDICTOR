# src/preprocess.py
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive cleaning: drop common unnecessary columns and fill na.
    """
    df = df.copy()
    for col in ["S.No", "Sno", "Date"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    df.fillna(0, inplace=True)
    return df

def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split into X and y and then into train/test.
    Expects columns: PM2.5, NO2, SO2, OZONE, AQI
    """
    X = df[["PM2.5", "NO2", "SO2", "OZONE"]]
    y = df["AQI"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def fit_scaler(X_train, X_test):
    """
    Fit a StandardScaler on X_train and transform both train & test.
    Returns X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
