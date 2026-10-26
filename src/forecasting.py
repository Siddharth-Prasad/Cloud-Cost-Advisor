"""Fleet cost forecasting: linear trend + weekday seasonality (NumPy only)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_total_cost(daily: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
    """
    daily: columns [date, total_cost]. Returns history + horizon forecast with a
    ~90% band. Model:  cost_t = a + b*t + weekday_effect(t) + noise
    """
    d = daily.sort_values("date").reset_index(drop=True)
    n = len(d)
    t = np.arange(n)

    def design(tt, dow):
        cols = [np.ones(len(tt)), tt.astype(float)]
        for k in range(1, 7):                       # weekday dummies (Mon = baseline)
            cols.append((dow == k).astype(float))
        return np.column_stack(cols)

    dow = d["date"].dt.dayofweek.values
    X = design(t, dow)
    beta, *_ = np.linalg.lstsq(X, d["total_cost"].values, rcond=None)
    fitted = X @ beta
    sigma = float(np.std(d["total_cost"].values - fitted, ddof=min(1, n - 1)))

    fut_dates = pd.date_range(d["date"].iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    tf = np.arange(n, n + horizon)
    pred = design(tf, fut_dates.dayofweek.values) @ beta
    widen = 1.645 * sigma * np.sqrt(1 + (np.arange(1, horizon + 1) / max(n, 1)))
    out_hist = pd.DataFrame({"date": d["date"], "actual": d["total_cost"], "forecast": fitted,
                             "lower": np.nan, "upper": np.nan, "kind": "history"})
    out_fut = pd.DataFrame({"date": fut_dates, "actual": np.nan, "forecast": pred,
                            "lower": np.maximum(pred - widen, 0), "upper": pred + widen,
                            "kind": "forecast"})
    return pd.concat([out_hist, out_fut], ignore_index=True)


def monthly_projection(fc: pd.DataFrame, days: int = 30) -> dict:
    f = fc[fc["kind"] == "forecast"].head(days)
    return {"expected": float(f["forecast"].sum()),
            "low": float(f["lower"].sum()), "high": float(f["upper"].sum())}


def trend_per_day(daily: pd.DataFrame) -> float:
    d = daily.sort_values("date")
    return float(np.polyfit(np.arange(len(d)), d["total_cost"].values, 1)[0])
