"""Cost anomaly detection: robust per-resource z-score + Isolation Forest."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 3.5,
                     contamination: float = 0.01) -> pd.DataFrame:
    """
    Flags a row when BOTH detectors agree it is unusual OR the robust z-score is
    extreme. Returns the flagged rows with a reason + estimated excess cost.
    """
    d = df.copy()
    grp = d.groupby("resource_id")["cost_usd"]
    med = grp.transform("median")
    mad = grp.transform(lambda s: np.median(np.abs(s - np.median(s))))
    mad = mad.replace(0, np.nan).fillna(d["cost_usd"].std() * 0.1 + 1e-6)
    d["robust_z"] = 0.6745 * (d["cost_usd"] - med) / mad
    d["baseline_cost"] = med

    feats = d[["cost_usd", "cpu_utilization", "memory_utilization", "running_hours"]]
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=150)
    d["iso_flag"] = iso.fit_predict(feats) == -1

    d["is_anomaly"] = ((d["robust_z"] > z_threshold) & (d["cost_usd"] > 1.5 * med)) | \
                      (d["iso_flag"] & (d["robust_z"] > z_threshold / 2))
    out = d[d["is_anomaly"]].copy()
    out["excess_cost"] = (out["cost_usd"] - out["baseline_cost"]).round(4)
    out["severity"] = pd.cut(out["robust_z"], [-np.inf, 6, 12, np.inf],
                             labels=["Medium", "High", "Critical"]).astype(str)
    out["reason"] = np.where(out["running_hours"] < 24, "Cost spike despite reduced hours",
                             "Cost spike vs resource baseline")
    cols = ["date", "resource_id", "instance_type", "environment", "cost_usd",
            "baseline_cost", "excess_cost", "robust_z", "severity", "reason"]
    return out[cols].sort_values("excess_cost", ascending=False).reset_index(drop=True)
