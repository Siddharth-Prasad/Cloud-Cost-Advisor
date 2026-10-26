"""
Rule-based cost optimization engine.

Works per RESOURCE (not per row) by aggregating the telemetry window, then
applies transparent, explainable rules:

  1. Idle            -> stop / terminate
  2. Underutilised   -> right-size to the cheapest instance that still fits
                        p95 CPU and memory demand (+25% headroom)
  3. Off-hours       -> schedule non-prod resources to office hours only
  4. Steady prod     -> purchase a commitment (savings plan / reserved)
  5. Storage review  -> large volumes attached to weak instances
  6. Overloaded      -> performance RISK, upsize (negative saving shown)

Savings are NOT double counted: rules 2 + 3 are combined on the new size,
rule 4 only applies when 1-3 did not.
"""
from __future__ import annotations

import pandas as pd

try:
    from . import config as C
except ImportError:                      # executed directly: python src/optimizer.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config as C

T = C.THRESHOLDS
PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2, "None": 3}


# ------------------------------------------------------------------ helpers
def summarise_resources(df: pd.DataFrame, lookback_days: int | None = None) -> pd.DataFrame:
    """Aggregate raw rows into one profile row per resource."""
    if lookback_days:
        df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=lookback_days - 1)]
    g = df.groupby("resource_id")
    prof = g.agg(
        instance_type=("instance_type", "last"),
        environment=("environment", "last"),
        region=("region", "last"),
        days=("date", "nunique"),
        avg_cpu=("cpu_utilization", "mean"),
        cpu_std=("cpu_utilization", "std"),
        avg_mem=("memory_utilization", "mean"),
        avg_hours=("running_hours", "mean"),
        storage_gb=("storage_gb", "last"),
        avg_daily_cost=("cost_usd", "mean"),
    )
    prof["p95_cpu"] = g["cpu_utilization"].quantile(0.95)
    prof["p95_mem"] = g["memory_utilization"].quantile(0.95)
    prof["cpu_std"] = prof["cpu_std"].fillna(0.0)
    prof["monthly_cost"] = prof["avg_daily_cost"] * C.DAYS_PER_MONTH
    return prof.reset_index()


def _compute_monthly(itype: str, region: str, hours: float) -> float:
    rate = C.HOURLY_RATE.get(itype, 0.0) * C.REGION_MULTIPLIER.get(region, 1.0)
    return rate * hours * C.DAYS_PER_MONTH


def find_right_size(itype: str, p95_cpu: float, p95_mem: float):
    """Cheapest catalog type that fits the observed peak demand (+headroom)."""
    cur = C.INSTANCE_CATALOG.get(itype)
    if cur is None:
        return None
    need_vcpu = cur["vcpu"] * (p95_cpu / 100.0) * C.SIZING_SAFETY_MARGIN
    need_mem = cur["mem_gb"] * (p95_mem / 100.0) * C.SIZING_SAFETY_MARGIN
    options = [(n, s) for n, s in C.INSTANCE_CATALOG.items()
               if s["vcpu"] >= need_vcpu and s["mem_gb"] >= need_mem and s["rate"] < cur["rate"]]
    if not options:
        return None
    return min(options, key=lambda x: x[1]["rate"])[0]


def _priority(saving: float, monthly_cost: float) -> str:
    if saving <= 0:
        return "None"
    share = saving / monthly_cost if monthly_cost else 0
    if saving >= 50 or share >= 0.5:
        return "High"
    if saving >= 15 or share >= 0.25:
        return "Medium"
    return "Low"


# --------------------------------------------------------------- main engine
def analyze_resource(r: pd.Series) -> dict:
    actions, rules = [], []
    saving = 0.0
    itype, region, env = r.instance_type, r.region, r.environment
    hours = float(r.avg_hours)
    compute_now = _compute_monthly(itype, region, hours)
    storage_cost = r.storage_gb * C.STORAGE_PRICE_GB_MONTH
    status, new_type, new_hours = "Optimal", itype, hours

    if r.avg_cpu < T["idle_cpu"] and r.avg_mem < T["idle_mem"]:
        status = "Idle"
        saving = r.monthly_cost * T["terminate_saving_pct"]
        rules.append("IDLE")
        actions.append(f"Stop/terminate {itype} (CPU {r.avg_cpu:.1f}%, memory {r.avg_mem:.1f}%). "
                       "Snapshot data first." if env != "prod" else
                       f"Verify ownership, then stop {itype} (CPU {r.avg_cpu:.1f}%).")
        new_type, new_hours = "(terminated)", 0.0
    else:
        new_compute = compute_now
        if r.avg_cpu < T["underused_cpu"] and r.avg_mem < T["underused_mem"]:
            status = "Underutilized"
            target = find_right_size(itype, r.p95_cpu, r.p95_mem)
            if target:
                new_type = target
                new_compute = _compute_monthly(target, region, new_hours)
                rules.append("RIGHT-SIZE")
                actions.append(f"Right-size {itype} to {target} "
                               f"(p95 CPU {r.p95_cpu:.0f}%, p95 mem {r.p95_mem:.0f}%).")
        if env in T["offhours_env"] and hours > T["offhours_running_hours"]:
            if status == "Optimal":
                status = "Off-hours waste"
            office_avg = T["office_hours_per_day"] * 5 / 7   # weekdays only
            new_hours = min(hours, office_avg)
            new_compute = _compute_monthly(new_type, region, new_hours)
            rules.append("SCHEDULE")
            actions.append(f"Schedule {env} resource to office hours "
                           f"({T['office_hours_per_day']}h/weekday, off at weekends).")
        saving = max(0.0, compute_now - new_compute)

        if not rules and env == "prod" and hours >= T["steady_prod_hours"] \
                and r.cpu_std <= T["steady_cpu_std"] \
                and T["underused_cpu"] <= r.avg_cpu <= T["overloaded_cpu"]:
            status = "Commitment candidate"
            saving = compute_now * T["commitment_discount"]
            rules.append("COMMIT")
            actions.append("Stable 24x7 production load: buy a 1-year savings plan / reserved "
                           f"instance (~{T['commitment_discount']:.0%} discount).")

        if r.avg_cpu > T["overloaded_cpu"]:
            status = "Overloaded"
            rules.append("RISK")
            bigger = [n for n, s in C.INSTANCE_CATALOG.items()
                      if s["tier"] == C.TYPE_TO_CODE.get(itype, 0) + 1]
            if bigger:
                new_type = bigger[0]
            actions.append(f"PERFORMANCE RISK: CPU {r.avg_cpu:.0f}% - upsize"
                           + (f" to {bigger[0]}." if bigger else ".") + " (cost will rise)")

    # storage review is additive (different cost component)
    if r.storage_gb >= T["storage_overprov_gb"] and r.avg_cpu < T["underused_cpu"]:
        s_save = storage_cost * 0.40
        saving += s_save
        rules.append("STORAGE")
        actions.append(f"Review {r.storage_gb:.0f} GB volume: clean up / move to cheaper tier "
                       f"(~${s_save:.2f}/mo).")

    # never claim more than the cost actually observed (guards against price-list mismatch)
    saving = round(float(min(saving, r.monthly_cost)), 2)
    risk = "RISK" in rules
    priority = "High" if risk else _priority(saving, r.monthly_cost)
    confidence = "High" if r.days >= 30 and r.cpu_std < 25 else ("Medium" if r.days >= 14 else "Low")
    return {
        "Resource ID": r.resource_id, "Environment": env, "Region": region,
        "Instance Type": itype, "Recommended Type": new_type,
        "Avg CPU (%)": round(r.avg_cpu, 1), "p95 CPU (%)": round(r.p95_cpu, 1),
        "Avg Memory (%)": round(r.avg_mem, 1), "Avg Hours/Day": round(hours, 1),
        "Monthly Cost ($)": round(r.monthly_cost, 2), "Status": status,
        "Rules Triggered": ", ".join(rules) if rules else "-",
        "Recommended Action": " ".join(actions) if actions else "No action required.",
        "Est. Monthly Savings ($)": saving,
        "Savings %": round(100 * saving / r.monthly_cost, 1) if r.monthly_cost else 0.0,
        "Priority": priority, "Confidence": confidence,
    }


def analyze_resource_efficiency(df: pd.DataFrame, lookback_days: int | None = None) -> pd.DataFrame:
    """Return one recommendation row per resource, highest savings first."""
    if df.empty:
        return pd.DataFrame()
    prof = summarise_resources(df, lookback_days)
    out = pd.DataFrame([analyze_resource(row) for _, row in prof.iterrows()])
    out["_p"] = out["Priority"].map(PRIORITY_ORDER)
    out = out.sort_values(["_p", "Est. Monthly Savings ($)"], ascending=[True, False]).drop(columns="_p")
    return out.reset_index(drop=True)


def efficiency_score(recs: pd.DataFrame) -> float:
    """0-100 fleet score: 100 = no identifiable waste."""
    if recs.empty or recs["Monthly Cost ($)"].sum() == 0:
        return 100.0
    waste = recs["Est. Monthly Savings ($)"].sum() / recs["Monthly Cost ($)"].sum()
    return round(max(0.0, 100 * (1 - waste)), 1)


def savings_by_category(recs: pd.DataFrame) -> pd.DataFrame:
    acts = recs[recs["Est. Monthly Savings ($)"] > 0]
    return (acts.groupby("Status")["Est. Monthly Savings ($)"].sum()
            .sort_values(ascending=False).reset_index())


def what_if(recs: pd.DataFrame, accept: list[str]) -> dict:
    """Savings if only the selected resource ids are optimised."""
    sel = recs[recs["Resource ID"].isin(accept)]
    cur = float(recs["Monthly Cost ($)"].sum())
    sav = float(sel["Est. Monthly Savings ($)"].sum())
    return {"current_monthly": cur, "savings": sav, "new_monthly": cur - sav,
            "annual_savings": sav * 12, "pct": 100 * sav / cur if cur else 0.0}


if __name__ == "__main__":
    # quick self-test with one obviously idle and one obviously busy resource
    import numpy as np
    d = pd.date_range("2026-01-01", periods=40)
    rows = []
    for rid, cpu, mem, env, it in [("i-idle", 2, 8, "dev", "m5.xlarge"),
                                   ("i-busy", 60, 60, "prod", "m5.large")]:
        for x in d:
            rows.append(dict(date=x, resource_id=rid, instance_type=it, region="ap-south-1",
                             environment=env, cpu_utilization=cpu + np.random.rand(),
                             memory_utilization=mem, running_hours=24, storage_gb=50,
                             network_gb=1, cost_usd=5.0))
    print(analyze_resource_efficiency(pd.DataFrame(rows))[
        ["Resource ID", "Status", "Recommended Action", "Est. Monthly Savings ($)"]])
