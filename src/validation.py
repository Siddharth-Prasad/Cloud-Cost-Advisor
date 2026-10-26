"""
Validation of the rule engine against KNOWN ground truth.

The synthetic generator plants a hidden behaviour in every resource
(idle, underused, overloaded, off-hours waste, steady production, healthy).
Here we run the optimizer blind and compare its verdict with that truth:
confusion matrix, precision / recall / F1, false-alarm rate, and how the
accuracy changes with the number of days of telemetry observed.

Honest scope: this measures whether the rules recover patterns they were
designed to find. It is NOT evidence of accuracy on real cloud accounts.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from . import config as C
from .data_preprocessing import clean_data
from .optimizer import analyze_resource_efficiency

EXPECTED = {
    "healthy": "Optimal", "idle": "Idle", "underused": "Underutilized",
    "overloaded": "Overloaded", "offhours_waste": "Off-hours waste",
    "steady_prod": "Commitment candidate",
}
LABELS = ["Optimal", "Idle", "Underutilized", "Off-hours waste", "Commitment candidate", "Overloaded"]


def _score(truth: pd.DataFrame, recs: pd.DataFrame) -> pd.DataFrame:
    m = truth.merge(recs[["Resource ID", "Status"]], left_on="resource_id",
                    right_on="Resource ID", how="left")
    m["Status"] = m["Status"].fillna("Optimal")
    return m


def perturb(clean: pd.DataFrame, sigma: float, seed: int) -> pd.DataFrame:
    """Stress test: shift each resource's CPU and memory readings by a random bias
    (std = sigma percentage points) to mimic monitoring error and mixed workloads."""
    if sigma <= 0:
        return clean
    rng = np.random.default_rng(seed + 1000)
    ids = clean["resource_id"].unique()
    bias_cpu = dict(zip(ids, rng.normal(0, sigma, len(ids))))
    bias_mem = dict(zip(ids, rng.normal(0, sigma, len(ids))))
    out = clean.copy()
    out["cpu_utilization"] = (out["cpu_utilization"] + out["resource_id"].map(bias_cpu)).clip(0, 100)
    out["memory_utilization"] = (out["memory_utilization"] + out["resource_id"].map(bias_mem)).clip(0, 100)
    return out


def run_benchmark(n_resources: int = 120, days: int = 60, seed: int = 7, noise: float = 8.0,
                  windows=(3, 7, 14, 30, 60), noise_levels=(0, 3, 6, 9, 12, 15, 20)) -> dict:
    if str(C.ROOT) not in sys.path:
        sys.path.insert(0, str(C.ROOT))
    from generate_data import build_dataset

    raw, truth = build_dataset(days=days, n_resources=n_resources, seed=seed)
    clean, _ = clean_data(raw)
    truth = truth.copy()
    truth["expected"] = truth["profile"].map(EXPECTED)

    noisy = perturb(clean, noise, seed)
    full = _score(truth, analyze_resource_efficiency(noisy))
    cm = confusion_matrix(full["expected"], full["Status"], labels=LABELS)
    confusion = pd.DataFrame(cm, index=LABELS, columns=LABELS)

    p, r, f, sup = precision_recall_fscore_support(
        full["expected"], full["Status"], labels=LABELS, zero_division=0)
    per_class = pd.DataFrame({"Pattern": LABELS, "Precision": p, "Recall": r, "F1": f, "Resources": sup})

    healthy = full[full["expected"] == "Optimal"]
    false_alarm = float((healthy["Status"] != "Optimal").mean()) if len(healthy) else 0.0

    curve = []
    for w in windows:
        if w > days:
            continue
        m = _score(truth, analyze_resource_efficiency(noisy, lookback_days=w))
        curve.append({"Days observed": w,
                      "Accuracy": accuracy_score(m["expected"], m["Status"]),
                      "Macro F1": f1_score(m["expected"], m["Status"], labels=LABELS,
                                           average="macro", zero_division=0)})
    noise_curve = []
    for sg in noise_levels:
        m = _score(truth, analyze_resource_efficiency(perturb(clean, sg, seed)))
        noise_curve.append({"Reading error (std, points)": sg,
                            "Accuracy": accuracy_score(m["expected"], m["Status"]),
                            "Macro F1": f1_score(m["expected"], m["Status"], labels=LABELS,
                                                 average="macro", zero_division=0)})
    return {
        "n_resources": n_resources, "days": days, "seed": seed, "noise": noise,
        "noise_curve": pd.DataFrame(noise_curve),
        "accuracy": float(accuracy_score(full["expected"], full["Status"])),
        "macro_f1": float(f1_score(full["expected"], full["Status"], labels=LABELS,
                                   average="macro", zero_division=0)),
        "false_alarm_rate": false_alarm,
        "confusion": confusion, "per_class": per_class, "curve": pd.DataFrame(curve),
    }
