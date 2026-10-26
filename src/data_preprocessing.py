"""Data loading, validation, cleaning and feature engineering."""
from __future__ import annotations

import io
from typing import Tuple, Union

import numpy as np
import pandas as pd

from . import config as C

MODEL_FEATURES = [
    "instance_type_code", "cpu_utilization", "memory_utilization",
    "running_hours", "storage_gb", "network_gb", "region_code",
    "env_code", "is_weekend", "day_of_week",
]


class DataValidationError(ValueError):
    """Raised when an uploaded dataset cannot be used."""


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Required: {', '.join(C.REQUIRED_COLUMNS)}")


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Clean a raw telemetry frame. Returns (clean_df, report)."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    validate_columns(df)
    report = {"rows_in": len(df)}

    for col, default in C.OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    num_cols = ["cpu_utilization", "memory_utilization", "running_hours",
                "storage_gb", "network_gb", "cost_usd"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)

    before = len(df)
    df = df.dropna(subset=["date", "resource_id", "instance_type", "cost_usd"])
    # missing sensor-style values are imputed per resource (median), not dropped
    report["missing_values_imputed"] = int(df[num_cols].isna().sum().sum())
    for col in ["cpu_utilization", "memory_utilization", "running_hours",
                "storage_gb", "network_gb"]:
        df[col] = df.groupby("resource_id")[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median()).fillna(0)
    report["rows_dropped_invalid"] = before - len(df)

    # validity ranges (clip rather than discard)
    df["cpu_utilization"] = df["cpu_utilization"].clip(0, 100)
    df["memory_utilization"] = df["memory_utilization"].clip(0, 100)
    df["running_hours"] = df["running_hours"].clip(0, 24)
    df = df[df["cost_usd"] >= 0]

    df["instance_type"] = df["instance_type"].astype(str).str.strip().str.lower()
    df["environment"] = df["environment"].astype(str).str.strip().str.lower()
    df["region"] = df["region"].astype(str).str.strip().str.lower()

    unknown = sorted(set(df["instance_type"]) - set(C.INSTANCE_CATALOG))
    report["unknown_instance_types"] = unknown

    df = add_features(df)
    df = df.sort_values(["date", "resource_id"]).reset_index(drop=True)
    report["rows_out"] = len(df)
    return df, report


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["instance_type_code"] = df["instance_type"].map(C.TYPE_TO_CODE).fillna(1).astype(int)
    df["region_code"] = df["region"].map(C.REGION_TO_CODE).fillna(0).astype(int)
    df["env_code"] = df["environment"].map(C.ENV_TO_CODE).fillna(0).astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


def load_and_clean_data(source: Union[str, "io.IOBase", None] = None,
                        return_report: bool = False):
    """Load a CSV path / file-like object and return a cleaned DataFrame."""
    source = source if source is not None else C.DATA_PATH
    raw = pd.read_csv(source)
    df, report = clean_data(raw)
    return (df, report) if return_report else df


def time_based_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Chronological split: the model never trains on the future."""
    df = df.sort_values("date")
    cutoff_idx = int(len(df) * (1 - test_fraction))
    cutoff_date = df.iloc[cutoff_idx]["date"]
    train = df[df["date"] < cutoff_date]
    test = df[df["date"] >= cutoff_date]
    return train, test


def daily_totals(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("date", as_index=False)["cost_usd"].sum().rename(
        columns={"cost_usd": "total_cost"})
