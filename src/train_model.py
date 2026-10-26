"""
Train, compare, evaluate and persist the cost-prediction model.

Run from the project root:   python src/train_model.py
Improvements over a single-model baseline:
  * chronological (time-aware) split -> no future leakage
  * three candidate models compared (Linear, Random Forest, Gradient Boosting)
  * MAE, RMSE, R2, MAPE + 5-fold TimeSeriesSplit cross-validation
  * feature importances & metrics persisted for the dashboard
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

try:                                    # works as `python src/train_model.py`
    from src import config as C
    from src.data_preprocessing import MODEL_FEATURES, load_and_clean_data, time_based_split
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config as C
    from src.data_preprocessing import MODEL_FEATURES, load_and_clean_data, time_based_split


def evaluate(y_true, y_pred) -> dict:
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": mape,
    }


def train_and_save_model(data_path=None, verbose=True) -> dict:
    df = load_and_clean_data(data_path)
    train, test = time_based_split(df, 0.2)
    X_train, y_train = train[MODEL_FEATURES], train["cost_usd"]
    X_test, y_test = test[MODEL_FEATURES], test["cost_usd"]

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                               random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                       max_depth=3, random_state=42),
    }

    results, fitted = {}, {}
    tscv = TimeSeriesSplit(n_splits=5)
    ordered = df.sort_values("date")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        m = evaluate(y_test.values, model.predict(X_test))
        cv = -cross_val_score(model, ordered[MODEL_FEATURES], ordered["cost_usd"],
                              cv=tscv, scoring="neg_mean_absolute_error")
        m["CV_MAE_mean"], m["CV_MAE_std"] = float(cv.mean()), float(cv.std())
        results[name], fitted[name] = m, model
        if verbose:
            print(f"{name:20s} MAE=${m['MAE']:.4f}  RMSE=${m['RMSE']:.4f}  "
                  f"R2={m['R2']:.3f}  CV-MAE={m['CV_MAE_mean']:.4f}")

    # The manual's chosen model is Random Forest; we still select the best by
    # test RMSE and record the decision so it can be defended in the viva.
    best_name = min(results, key=lambda k: results[k]["RMSE"])
    best = fitted[best_name]

    importances = {}
    if hasattr(best, "feature_importances_"):
        importances = dict(zip(MODEL_FEATURES, map(float, best.feature_importances_)))

    # Refit the winner on ALL data for production forecasting
    final = type(best)(**best.get_params()).fit(df[MODEL_FEATURES], df["cost_usd"])

    # residual quantiles -> prediction intervals in the UI
    resid = y_test.values - best.predict(X_test)
    interval = {"lo": float(np.quantile(resid, 0.05)), "hi": float(np.quantile(resid, 0.95))}

    C.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": MODEL_FEATURES, "name": best_name}, C.MODEL_PATH)
    summary = {
        "best_model": best_name, "results": results, "feature_importance": importances,
        "interval_90": interval, "n_train": int(len(train)), "n_test": int(len(test)),
        "test_start": str(test["date"].min().date()), "train_end": str(train["date"].max().date()),
    }
    C.METRICS_PATH.write_text(json.dumps(summary, indent=2))
    if verbose:
        print(f"\nBest model: {best_name}")
        print(f"Saved model   -> {C.MODEL_PATH}")
        print(f"Saved metrics -> {C.METRICS_PATH}")
    return summary


if __name__ == "__main__":
    train_and_save_model()
