# Viva & Submission Guide

## Abstract (copy into your report)
Unmonitored and over-provisioned cloud resources cause significant financial waste. This project presents a Machine Learning-Based Cloud Cost Optimization Advisor that ingests cloud telemetry, predicts cost with regression models (Linear Regression, Random Forest, Gradient Boosting compared on a chronological hold-out), forecasts future spend, detects cost anomalies (robust z-score + Isolation Forest) and applies an explainable rule engine to recommend idle-resource shutdown, p95-based right-sizing, off-hours scheduling and commitment purchases. A Streamlit dashboard presents analytics, a what-if savings simulator and exportable reports. The system runs fully offline on free tools.

## 5-minute demo script
1. Overview — efficiency score, savings, top 5 opportunities.
2. Cost analysis → Resource drill-down on an idle resource.
3. Forecast — show band and "spend after optimisation".
4. Cost predictor — change instance type/hours and show the prediction and price-list sanity check.
5. Anomalies — move the sensitivity slider.
6. Recommendations — filter High priority, download the HTML report.
7. Savings simulator — accept High priority only, show annual saving in ₹.
8. Model Performance — explain model comparison.

## Viva questions
1. **Why compare three models?** Linear Regression is a baseline; tree ensembles capture non-linear interactions (instance size × hours × environment). Winner chosen by test RMSE, not by assumption.
2. **Data leakage — how avoided?** Chronological split (train always earlier than test), features contain no cost-derived columns, and the final model is refit on all data only after evaluation.
3. **Why MAE and RMSE?** MAE = average dollar error; RMSE penalises large misses. RMSE > MAE here because of rare injected cost spikes.
4. **Why is R² not ~1?** The data contains random spikes (≈0.6% of rows, 2–4.5× cost) plus noise; they are unpredictable by design (they are what the anomaly detector finds).
5. **How does right-sizing work?** Needed vCPU/memory = instance size × p95 utilisation × 1.25 headroom; pick the cheapest catalog type that satisfies both and is cheaper than the current one.
6. **Why p95, not average?** Sizing for the average would cause performance incidents at peak.
7. **How are savings estimated, and are they double counted?** From the price list (rate × hours × 30). Rules are combined per resource (e.g. right-size + schedule on the new size); commitment discount applies only when no other rule fired; savings are capped at observed cost.
8. **Why Joblib instead of Pickle?** Efficient for large NumPy arrays in scikit-learn models.
9. **How does Streamlit work?** Re-runs the script top-to-bottom on every interaction; `st.cache_data` avoids recomputing heavy steps.
10. **How does the forecast work?** OLS fit of linear trend + day-of-week dummies on daily totals; interval from residual spread widening with horizon.
11. **How does anomaly detection work?** Per-resource median/MAD robust z-score catches spikes relative to that resource's own baseline; Isolation Forest adds multivariate outlier evidence.
12. **Limitations?** Synthetic data, illustrative prices, rule thresholds are heuristics, forecast ignores events/releases, no live cloud API.
13. **Future work?** AWS Cost Explorer / CloudWatch integration, LSTM/Prophet forecasting, auto-remediation with approval workflow, multi-cloud pricing.

## Known honesty points (say these confidently)
* Dataset is synthetic with planted waste profiles — the engine rediscovering them validates the rules, not real-world accuracy.
* Prices are illustrative, not an official quote.
