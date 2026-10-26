"""
Central configuration: instance catalog, pricing and optimisation thresholds.

All prices are illustrative on-demand USD/hour values modelled on public
cloud price lists (rounded). They are used for ACADEMIC DEMONSTRATION only.
"""
from pathlib import Path

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cloud_usage.csv"
MODEL_PATH = ROOT / "models" / "cloud_cost_model.pkl"
METRICS_PATH = ROOT / "models" / "model_metrics.json"
REPORT_DIR = ROOT / "reports"

# ---------------------------------------------------------- instance catalog
# tier (1 = smallest) is also the ordinal feature used by the ML model.
INSTANCE_CATALOG = {
    "t3.micro":   {"tier": 1, "vcpu": 2, "mem_gb": 1,  "rate": 0.0104},
    "t3.medium":  {"tier": 2, "vcpu": 2, "mem_gb": 4,  "rate": 0.0416},
    "m5.large":   {"tier": 3, "vcpu": 2, "mem_gb": 8,  "rate": 0.0960},
    "c5.xlarge":  {"tier": 4, "vcpu": 4, "mem_gb": 8,  "rate": 0.1700},
    "m5.xlarge":  {"tier": 5, "vcpu": 4, "mem_gb": 16, "rate": 0.1920},
    "m5.2xlarge": {"tier": 6, "vcpu": 8, "mem_gb": 32, "rate": 0.3840},
}
INSTANCE_TYPES = list(INSTANCE_CATALOG)
TYPE_TO_CODE = {k: v["tier"] for k, v in INSTANCE_CATALOG.items()}
CODE_TO_TYPE = {v: k for k, v in TYPE_TO_CODE.items()}
HOURLY_RATE = {k: v["rate"] for k, v in INSTANCE_CATALOG.items()}

STORAGE_PRICE_GB_MONTH = 0.10     # USD per GB-month
NETWORK_PRICE_GB = 0.09           # USD per GB egress
DAYS_PER_MONTH = 30

REGIONS = ["ap-south-1", "us-east-1", "eu-west-1"]
REGION_MULTIPLIER = {"ap-south-1": 1.00, "us-east-1": 0.95, "eu-west-1": 1.08}
REGION_TO_CODE = {r: i for i, r in enumerate(REGIONS)}

ENVIRONMENTS = ["prod", "staging", "dev", "test"]
ENV_TO_CODE = {e: i for i, e in enumerate(ENVIRONMENTS)}

# ---------------------------------------------------------------- rule engine
THRESHOLDS = {
    "idle_cpu": 5.0,               # avg CPU % below which a resource is idle
    "idle_mem": 15.0,
    "underused_cpu": 20.0,         # avg CPU % for "underutilised"
    "underused_mem": 30.0,
    "overloaded_cpu": 85.0,        # avg CPU % -> risk, suggest upsizing
    "p95_headroom_cpu": 60.0,      # right-size only if p95 CPU stays below this
    "offhours_env": ["dev", "test", "staging"],
    "offhours_running_hours": 20,  # non-prod running > this many h/day = waste
    "office_hours_per_day": 10,    # target schedule (e.g. 09:00-19:00)
    "steady_prod_hours": 22,       # reserved / savings-plan candidate
    "steady_cpu_std": 18.0,        # stable load only
    "commitment_discount": 0.30,   # 1-yr savings plan (illustrative)
    "storage_overprov_gb": 300,    # big volume attached to a weak instance
    "terminate_saving_pct": 0.90,  # share of cost saved when terminated
}

SIZING_SAFETY_MARGIN = 1.25  # keep 25% headroom when right-sizing

REQUIRED_COLUMNS = [
    "date", "resource_id", "instance_type", "cpu_utilization",
    "memory_utilization", "running_hours", "storage_gb", "cost_usd",
]
OPTIONAL_DEFAULTS = {          # filled in when a user uploads a simpler CSV
    "region": "ap-south-1",
    "environment": "prod",
    "network_gb": 0.0,
}
