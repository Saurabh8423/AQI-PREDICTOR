# src/data_loader.py
import pandas as pd
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset CSV into a pandas DataFrame.
    """
    df = pd.read_csv(r"./data/aqi_dataset.csv")
    return df
