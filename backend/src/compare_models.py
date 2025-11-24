# src/compare_models.py
from typing import Dict
import pandas as pd

def summarize_results(results: Dict[str, dict]) -> pd.DataFrame:
    """
    results: dict mapping model name -> {'mae':..., 'r2':...}
    returns sorted DataFrame by R2 desc
    """
    rows = []
    for name, metrics in results.items():
        rows.append({"Model": name, "MAE": metrics["mae"], "R2": metrics["r2"]})
    df = pd.DataFrame(rows)
    df = df.sort_values(by="R2", ascending=False).reset_index(drop=True)
    return df
