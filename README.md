# ☁️ Machine Learning-Based Cloud Cost Optimization Advisor
MCA project · Python 3.11+ · pandas · scikit-learn · Streamlit · Plotly · 100% free & local

## Quick start (Windows 11 + VS Code)
```bat
setup.bat      :: creates venv, installs packages, generates data, trains model
run.bat        :: launches the dashboard at http://localhost:8501
```
Manual steps:
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
python src\train_model.py
python src\optimizer.py           :: optional self-test
python -m unittest discover -s tests -v
streamlit run app.py
```

## Features
| Area | What it does |
|---|---|
| Data pipeline | Validation, de-duplication, per-resource median imputation, range clipping, feature engineering, data-quality report |
| ML models | Linear Regression vs Random Forest vs Gradient Boosting, chronological split, TimeSeriesSplit CV, MAE/RMSE/R²/MAPE, 90% prediction interval |
| Optimization engine | Idle, underutilised (p95 right-sizing), off-hours scheduling, commitment/savings-plan, storage review, overload risk; no double counting |
| Forecasting | 30–90 day spend forecast (trend + weekday seasonality) with confidence band |
| Anomaly detection | Robust z-score + Isolation Forest, severity levels, excess-cost estimate |
| Dashboard | 8 pages with grouped sidebar navigation, global filters, resource drill-down, what-if savings simulator (USD → INR), efficiency score |
| Bring your own data | Upload any CSV with the required columns |
| Exports | CSV, Excel, standalone HTML executive report |
| Quality | 15 unit tests, central `config.py` for all thresholds and prices |

## CSV format
Required: `date, resource_id, instance_type, cpu_utilization, memory_utilization, running_hours, storage_gb, cost_usd`
Optional: `region, environment (prod/staging/dev/test), network_gb`.
Supported instance types: t3.micro, t3.medium, m5.large, c5.xlarge, m5.xlarge, m5.2xlarge (edit `src/config.py` to add more).

## Folder structure
```
CloudCostAdvisor/
├── app.py                  Streamlit dashboard
├── generate_data.py        synthetic telemetry generator
├── assets/                 logo files used by the sidebar
├── requirements.txt  setup.bat  run.bat
├── data/cloud_usage.csv    generated
├── models/                 cloud_cost_model.pkl, model_metrics.json (generated)
├── src/  config.py data_preprocessing.py train_model.py optimizer.py
│         forecasting.py anomaly.py reports.py
├── tests/test_project.py
└── docs/VIVA_GUIDE.md      abstract, viva Q&A, demo script
```

## Notes
* All data is **synthetic** and prices are illustrative — for academic demonstration only.
* The tool is **advisory**: it never modifies cloud resources.
* Needs Streamlit 1.46 or newer (dark sidebar theme and grouped navigation). Upgrade with: pip install --upgrade streamlit
