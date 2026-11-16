# Fraud-Risk Intelligence Pipeline: README (v2)

Calibrated fraud probability + Approve / Review / Block decision + SHAP reason codes, built on `transaction_fraud_risk_advanced_dataset.csv` (3,000 transactions, 364 fraud, 12.1%).

## Deliverables
| File | Content |
|---|---|
| `01_fraud_risk_analysis.ipynb` | Executed notebook, sections A-K: audit, forensics, EDA, **89 engineered features in 11 groups** (univariate evaluation, correlation, target-encoding, group ablation, selection test), 7 models (LogReg, RandomForest, ExtraTrees, HistGradBoosting, LightGBM, XGBoost, soft-voting ensemble) with randomised HPO, learning curves, imbalance study, calibration, cost framework and thresholds, hold-out evaluation, operations simulation, SHAP, permutation importance, stress tests, monitoring |
| `02_processed_dataset.csv` | Raw + engineered features for all rows with a `split` column (dev/test). Statistics were fitted on the dev split only |
| `03_fraud_model_pipeline.joblib` | Preprocessing + feature engineering + ExtraTrees pipeline (uncalibrated), fitted on the dev set |
| `04_calibrated_model.joblib` | `CalibratedClassifierCV` (Platt) around the pipeline: production candidate |
| `05_threshold_cost_analysis.csv` | Long-format tables (column `table`): threshold curves, cost-scenario sensitivity, threshold grid, model x scenario costs, policy comparison |
| `06_model_card.pdf` | Model governance document (3 pages) |
| `07_business_fraud_strategy.pdf` | Management and technical report (38 pages, no headers or footers) |
| `08_README.md` | This file |
| Extras | `fraud_lib.py` (required to load the joblibs), `score_transactions.py` (scoring CLI), `decision_policy.json` (thresholds + costs), `feature_dictionary.csv`, `feature_univariate_evaluation.csv`, `feature_group_ablation.csv`, `permutation_importance_raw_fields.csv`, `local_explanations_reason_codes.csv`, `hpo_log.csv`, `data_quality_report.csv`, `results.json` (every number used in the reports), `figures/` (F01-F33 PNG) |

## Environment
Python 3.13; numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.4.1, imbalanced-learn 0.14.2, shap 0.54.1, matplotlib, seaborn, scipy, joblib, reportlab, nbformat/nbclient.
```
pip install numpy pandas scikit-learn lightgbm xgboost imbalanced-learn shap matplotlib seaborn scipy joblib jupyter reportlab
```

## How to run
1. Keep `fraud_lib.py`, the CSV and the notebook in one folder.
2. `jupyter nbconvert --to notebook --execute --inplace 01_fraud_risk_analysis.ipynb` (about 6 minutes on a few CPU cores). It regenerates figures, CSVs, joblibs and `results.json`.
3. Score new data: `python score_transactions.py new_transactions.csv scored.csv` (probability, decision, top-3 SHAP reason codes for flagged rows).
4. In your own code: `import fraud_lib` first (it defines `FeatureEngineer`), then `joblib.load(...)`; `predict_proba(df_raw_features)[:,1]`, then apply the thresholds in `decision_policy.json`.

## Reproducibility and leakage controls
* Seed **42** for the split, CV folds, searches, samplers, bootstraps and stress tests. The same 5 stratified folds are used for every model.
* 80/20 stratified split: development 2,400 rows, **locked test 600 rows**, used only for final evaluation, operations simulation, bootstrap intervals and stress tests.
* All learned steps (imputation, scaling, one-hot, group means / percentiles / bin edges, cross-fitted target encoding, any resampling, feature selection) live inside pipelines and are fitted on training folds only.
* Calibration is cross-fitted; thresholds are optimised on nested out-of-fold calibrated probabilities of the dev set.
* No observations were removed; `transaction_id` is excluded.

## Key assumptions (hypothetical, sensitivity-tested)
* Costs (USD): missed fraud = amount + $15; review = $20 + $3 friction, analyst stops 90% of fraud; false block = $40 + 2% of amount.
* Thresholds: Review at p >= 0.14, Block at p >= 0.35 (genuine three-way policy; each band >= 1% of volume). Replace the costs with finance-approved values and re-optimise.
* Volume scaling to 100,000 transactions/month assumes the test mix is representative.
* `day_of_week` 0 = Monday; risk scores, chargebacks and velocity windows are point-in-time values. No timestamp exists, so no out-of-time validation.

## Known limitations
Modest discrimination (test ROC-AUC 0.670, PR-AUC 0.189 vs no-skill 0.122); 12% prevalence is unrealistic for production; about 360 fraud cases; the Block band blocked no fraud on the test set (launch Approve / Review first); the model uses a new_device vs device_age_days inconsistency; thresholds are cost-sensitive. Re-calibrate on production prevalence, pilot in shadow mode, and review fairness (country / tier) before use.
