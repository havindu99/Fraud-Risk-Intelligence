"""Score new transactions with the calibrated model and apply the Approve / Review / Block policy.
Usage: python score_transactions.py input.csv output.csv      (input has the 23 raw feature columns; transaction_id optional)
Reason codes come from SHAP on the uncalibrated pipeline (03) and explain the MODEL, not causes."""
import sys, json, numpy as np, pandas as pd, joblib, shap
import fraud_lib  # needed to unpickle FeatureEngineer
from fraud_lib import feature_names

pol = json.load(open("decision_policy.json"))
cal = joblib.load("04_calibrated_model.joblib"); base = joblib.load("03_fraud_model_pipeline.joblib")
df = pd.read_csv(sys.argv[1]); ids = df["transaction_id"] if "transaction_id" in df else pd.Series(range(len(df)))
X = df[pol["features_raw"]]
p = cal.predict_proba(X)[:, 1]
t = pol["thresholds"]; act = pol["actions"]
dec = np.where(p >= t["block"], 2, np.where(p >= t["review"], 1, 0))
out = pd.DataFrame({"transaction_id": ids, "fraud_probability": p.round(4), "decision": [act[str(d)] for d in dec]})
flag = dec > 0; out["reason_codes"] = ""
if flag.any():
    prep = base.named_steps["prep"]; names = feature_names(base)
    Xt = pd.DataFrame(prep.transform(X[flag]), columns=names)
    v = shap.TreeExplainer(base.named_steps["model"]).shap_values(Xt, check_additivity=False)
    v = np.asarray(v[1]) if isinstance(v, list) else (np.asarray(v)[..., 1] if np.ndim(v) == 3 else np.asarray(v))
    out.loc[flag, "reason_codes"] = ["; ".join(f"{names[j]}={Xt.iloc[i, j]:.2f}" for j in np.argsort(-v[i])[:3]) for i in range(len(Xt))]
out.to_csv(sys.argv[2], index=False); print(out.decision.value_counts().to_dict())
