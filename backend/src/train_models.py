# src/train_models.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from typing import Any

def train_random_forest(X_train, y_train) -> Any:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm_grid(X_train, y_train) -> Any:
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.1, 0.01],
        "epsilon": [0.1, 0.2],
    }
    grid = GridSearchCV(SVR(kernel="rbf"), param_grid, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_decision_tree(X_train, y_train) -> Any:
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_linear_regression(X_train, y_train) -> Any:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_knn_grid(X_train, y_train) -> Any:
    param_grid = {
        "n_neighbors": [2, 3, 5],
        "weights": ["uniform", "distance"],
    }
    grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_
