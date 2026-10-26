"""
Generate a realistic synthetic cloud-telemetry dataset (daily granularity).

Run:  python generate_data.py [--days 120] [--resources 30] [--seed 42]

Every resource is assigned a hidden "behaviour profile" so the optimizer has
genuine waste to discover:
  healthy | idle | underused | overloaded | offhours_waste | steady_prod
The dataset is SYNTHETIC and for academic demonstration only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config as C  # noqa: E402

PROFILES = ["healthy", "idle", "underused", "overloaded", "offhours_waste", "steady_prod"]
PROFILE_P = [0.25, 0.10, 0.18, 0.07, 0.20, 0.20]


def build_dataset(days=120, n_resources=30, seed=42):
    """Return (telemetry_df, ground_truth_df). Ground truth lists the planted behaviour of each resource."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days, freq="D")

    # ---- build the fleet
    fleet = []
    for i in range(n_resources):
        profile = rng.choice(PROFILES, p=PROFILE_P)
        env = {
            "steady_prod": "prod", "overloaded": "prod", "healthy": rng.choice(["prod", "staging"]),
            "idle": rng.choice(["dev", "test"]), "underused": rng.choice(["prod", "staging", "dev"]),
            "offhours_waste": rng.choice(["dev", "test", "staging"]),
        }[profile]
        # oversized instances for waste profiles
        if profile in ("underused", "idle"):
            itype = rng.choice(["m5.large", "c5.xlarge", "m5.xlarge", "m5.2xlarge"])
        elif profile == "overloaded":
            itype = rng.choice(["t3.micro", "t3.medium"])
        else:
            itype = rng.choice(C.INSTANCE_TYPES)
        fleet.append({
            "resource_id": f"i-{rng.integers(0x10000, 0xFFFFF):05x}{i:02d}",
            "profile": profile, "environment": env, "instance_type": str(itype),
            "region": str(rng.choice(C.REGIONS)),
            "storage_gb": int(rng.choice([20, 50, 100, 250, 500, 1000])),
            "trend": rng.normal(0.0004, 0.0004),      # slow organic growth per day
        })

    rows = []
    for f in fleet:
        p, t = f["profile"], f["instance_type"]
        rate = C.HOURLY_RATE[t] * C.REGION_MULTIPLIER[f["region"]]
        for d_idx, date in enumerate(dates):
            weekend = date.dayofweek >= 5
            growth = 1 + f["trend"] * d_idx

            if p == "idle":
                cpu = rng.uniform(0.5, 4.5); mem = rng.uniform(5, 14); hours = 24.0
            elif p == "underused":
                cpu = rng.uniform(6, 19); mem = rng.uniform(10, 28); hours = 24.0
            elif p == "overloaded":
                cpu = rng.uniform(82, 99); mem = rng.uniform(75, 96); hours = 24.0
            elif p == "offhours_waste":
                cpu = rng.uniform(25, 60) if not weekend else rng.uniform(2, 8)
                mem = rng.uniform(30, 60); hours = 24.0
            elif p == "steady_prod":
                cpu = rng.normal(55, 6); mem = rng.normal(60, 5); hours = 24.0
            else:  # healthy - business-hours workload, sometimes auto-stopped
                cpu = rng.uniform(35, 75) * (0.7 if weekend else 1.0)
                mem = rng.uniform(40, 75)
                hours = float(rng.choice([24, 24, 20, 16, 12]))

            cpu = float(np.clip(cpu * growth, 0.5, 100)); mem = float(np.clip(mem, 2, 100))
            network = float(max(0.0, rng.gamma(2.0, 1.5 + cpu / 25)))
            compute = rate * hours
            storage = f["storage_gb"] * C.STORAGE_PRICE_GB_MONTH / C.DAYS_PER_MONTH
            net = network * C.NETWORK_PRICE_GB
            # occasional cost spike (anomaly) ~0.6% of rows
            spike = rng.uniform(2.0, 4.5) if rng.random() < 0.006 else 1.0
            cost = (compute + storage + net) * spike * (1 + rng.normal(0, 0.02))
            rows.append([date.strftime("%Y-%m-%d"), f["resource_id"], t, f["region"],
                         f["environment"], round(cpu, 2), round(mem, 2), hours,
                         f["storage_gb"], round(network, 2), round(max(cost, 0.001), 4)])

    df = pd.DataFrame(rows, columns=[
        "date", "resource_id", "instance_type", "region", "environment",
        "cpu_utilization", "memory_utilization", "running_hours",
        "storage_gb", "network_gb", "cost_usd"])

    # inject a few missing values / duplicates so preprocessing is meaningful
    idx = rng.choice(len(df), size=max(3, len(df) // 800), replace=False)
    df.loc[idx, "cpu_utilization"] = np.nan
    df = pd.concat([df, df.sample(3, random_state=seed)], ignore_index=True)
    df = df.sort_values(["date", "resource_id"]).reset_index(drop=True)

    truth = pd.DataFrame(fleet)[["resource_id", "profile", "environment"]]
    return df, truth


def generate_synthetic_data(days=120, n_resources=30, seed=42, out_path=C.DATA_PATH):
    df, truth = build_dataset(days, n_resources, seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset generated: {out_path}  ({len(df)} rows, "
          f"{n_resources} resources, {days} days)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--resources", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    generate_synthetic_data(a.days, a.resources, a.seed)
