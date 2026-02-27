from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


@dataclass
class TrainResult:
    model: object
    metrics: dict
    feature_importance: dict


def _build_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(n_estimators=200, random_state=42)
        except Exception:
            return RandomForestRegressor(n_estimators=300, random_state=42)
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(n_estimators=300, random_state=42, verbosity=0)
        except Exception:
            return RandomForestRegressor(n_estimators=300, random_state=42)
    return RandomForestRegressor(n_estimators=300, random_state=42)


def train(features: pd.DataFrame, model_type: str = "sklearn", target_col: str = "target") -> TrainResult:
    """Analysts can edit this function to tune model features/training logic."""
    excluded = {"timestamp", target_col, "symbol"}
    feature_cols = [c for c in features.columns if c not in excluded and np.issubdtype(features[c].dtype, np.number)]
    X = features[feature_cols]
    y = features[target_col]

    split = int(len(features) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = _build_model(model_type)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_type": model_type,
    }

    if hasattr(model, "feature_importances_"):
        importance = dict(zip(feature_cols, map(float, model.feature_importances_)))
    else:
        importance = {c: 0.0 for c in feature_cols}

    return TrainResult(model=model, metrics=metrics, feature_importance=importance)
