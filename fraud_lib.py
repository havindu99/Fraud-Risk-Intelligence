"""
fraud_lib.py - reusable, importable building blocks for the fraud-risk decision engine.

Everything that LEARNS from data (group statistics, imputation, scaling, encoding, resampling)
lives inside scikit-learn / imbalanced-learn pipelines so that it is fitted on training folds only.
This module must be importable when the .joblib pipelines are loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "fraud"
ID_COL = "transaction_id"

NUM_RAW = [
    "hour", "day_of_week", "transaction_amount", "distance_from_usual_location_km",
    "transactions_last_1h", "transactions_last_24h", "account_age_days", "failed_logins_24h",
    "prior_chargebacks", "device_age_days", "merchant_risk_score", "ip_risk_score",
    "email_age_days", "login_to_transaction_minutes", "new_device", "billing_address_mismatch",
]
CAT_RAW = ["card_presence", "channel", "country", "device_type", "merchant_category",
           "authentication_level", "customer_tier"]

# (name, group, definition / rationale, needs_fitted_state)
FE_SPEC = [
    # ---- transform: tame skew / rank-normalise ------------------------------------------------
    ("log_amount", "transform", "log1p(amount): amounts are heavy-tailed (max 5.5k vs median 88).", False),
    ("log_distance_km", "transform", "log1p(distance from usual location).", False),
    ("log_account_age", "transform", "log1p(account age in days): early-life accounts are riskier, effect is strongest at small ages.", False),
    ("log_device_age", "transform", "log1p(device age in days).", False),
    ("log_email_age", "transform", "log1p(email age in days).", False),
    ("log_login_minutes", "transform", "log1p(minutes between login and transaction).", False),
    ("sqrt_tx24h", "transform", "sqrt(transactions in 24h): dampens the extreme counts (max 113).", False),
    ("amount_pct_rank", "transform", "Percentile rank of the amount in the training distribution (robust to outliers).", True),
    ("distance_pct_rank", "transform", "Percentile rank of the distance in the training distribution.", True),
    # ---- velocity -------------------------------------------------------------------------------
    ("velocity_ratio_1h_24h", "velocity", "txns_1h / (txns_24h + 1): share of the day's activity that falls in the last hour (burst).", False),
    ("tx_prior_23h", "velocity", "txns_24h - txns_1h: activity before the current hour (baseline level).", False),
    ("hourly_rate_vs_daily", "velocity", "txns_1h / ((txns_24h + 1) / 24): current hourly rate relative to the 24h average rate (acceleration).", False),
    ("burst_flag", "velocity", "1 if 4 or more transactions in the last hour.", False),
    ("failed_logins_per_tx24", "velocity", "failed logins / (txns_24h + 1): authentication friction per transaction.", False),
    ("amount_x_tx1h", "velocity", "amount x txns_1h: money moved in a burst (exposure proxy).", False),
    # ---- amount normalisation -------------------------------------------------------------------
    ("amount_z_by_merchant_cat", "amount_norm", "z-score of log amount within the merchant category (is this big FOR a restaurant / jeweller?).", True),
    ("amount_z_by_channel", "amount_norm", "z-score of log amount within the channel.", True),
    ("amount_to_tier_median", "amount_norm", "amount / median amount of the customer tier.", True),
    ("amount_per_account_day", "amount_norm", "amount / (account age + 1): large amounts on young accounts.", False),
    ("amount_to_device_age", "amount_norm", "amount / (device age + 1).", False),
    ("high_amount_p90", "amount_norm", "1 if amount above the training 90th percentile.", True),
    ("high_amount_p95", "amount_norm", "1 if amount above the training 95th percentile.", True),
    # ---- risk combinations ----------------------------------------------------------------------
    ("combined_risk_mean", "risk", "mean(merchant risk, ip risk).", False),
    ("combined_risk_max", "risk", "max(merchant risk, ip risk): the worst of the two.", False),
    ("risk_product", "risk", "merchant risk x ip risk: both elevated at once.", False),
    ("risk_gap", "risk", "|merchant risk - ip risk|: disagreement between the two scores.", False),
    ("ip_risk_sq", "risk", "ip risk squared (convex effect).", False),
    ("merchant_risk_sq", "risk", "merchant risk squared (convex effect).", False),
    ("both_risk_high", "risk", "1 if both risk scores above their training 75th percentile.", True),
    ("red_flag_count", "risk", "Number of red flags: new device, address mismatch, prior chargeback, failed login, high IP risk (p90), high merchant risk (p90), fast login, account < 90d, weak auth & card-not-present, far distance (p90), high amount (p90).", True),
    # ---- time windows ---------------------------------------------------------------------------
    ("is_night", "time", "1 if hour 0-5.", False),
    ("is_weekend", "time", "1 if day_of_week >= 5 (ASSUMING 0 = Monday).", False),
    ("is_business_hours", "time", "1 if weekday and hour 9-17.", False),
    ("hour_sin", "time", "sin(2*pi*hour/24): cyclical hour.", False),
    ("hour_cos", "time", "cos(2*pi*hour/24).", False),
    ("dow_sin", "time", "sin(2*pi*dow/7): cyclical weekday.", False),
    ("dow_cos", "time", "cos(2*pi*dow/7).", False),
    ("time_block", "time", "Ordinal block: 0 night (0-5), 1 morning (6-11), 2 afternoon (12-17), 3 evening (18-23).", False),
    ("night_x_cnp", "time", "night x card-not-present.", False),
    # ---- behavioural deviation ------------------------------------------------------------------
    ("tx24h_dev_from_tier_median", "behaviour", "txns_24h minus median of the customer tier.", True),
    ("tx24h_dev_from_channel_median", "behaviour", "txns_24h minus median of the channel.", True),
    ("distance_z_by_country", "behaviour", "z-score of log distance within the country (far for a Sri Lankan customer vs far for a traveller).", True),
    ("login_z", "behaviour", "z-score of log login-to-transaction time vs training.", True),
    ("fast_login_flag", "behaviour", "1 if login->transaction time below the training 10th percentile (automation / takeover).", True),
    ("device_to_account_age_ratio", "behaviour", "device age / (account age + 1).", False),
    ("email_to_account_age_ratio", "behaviour", "email age / (account age + 1).", False),
    ("device_older_than_account", "behaviour", "1 if device age > account age.", False),
    ("new_device_inconsistent", "behaviour", "1 if new_device = 1 but device age > 30 days (data inconsistency kept as a signal).", False),
    # ---- interactions ---------------------------------------------------------------------------
    ("distance_x_new_device", "interaction", "log distance x new_device.", False),
    ("amount_x_merchant_risk", "interaction", "amount x merchant risk.", False),
    ("amount_x_ip_risk", "interaction", "amount x ip risk.", False),
    ("failed_logins_x_new_device", "interaction", "failed logins x new_device.", False),
    ("velocity_x_new_device", "interaction", "txns_1h x new_device.", False),
    ("mismatch_x_new_device", "interaction", "billing mismatch x new_device.", False),
    ("chargeback_x_new_device", "interaction", "prior chargebacks x new_device.", False),
    ("weak_auth_cnp", "interaction", "authentication not Strong (Standard / Missing) AND card not present.", False),
    ("new_device_x_cnp", "interaction", "new_device x card-not-present.", False),
    ("mismatch_x_cnp", "interaction", "billing mismatch x card-not-present.", False),
    ("far_x_high_amount", "interaction", "distance above training p75 AND amount above training p75.", True),
    # ---- binary flags ---------------------------------------------------------------------------
    ("new_account_flag", "flags", "account younger than 90 days.", False),
    ("very_new_account", "flags", "account younger than 30 days.", False),
    ("prior_chargeback_flag", "flags", "any prior chargeback.", False),
    ("multi_chargeback", "flags", "two or more prior chargebacks.", False),
    ("failed_login_flag", "flags", "at least one failed login in 24h.", False),
    ("many_failed_logins", "flags", "two or more failed logins in 24h.", False),
    ("young_device", "flags", "device younger than 30 days.", False),
    ("far_distance_flag", "flags", "distance above training p90.", True),
    ("high_ip_flag", "flags", "ip risk above training p90.", True),
    ("high_merchant_flag", "flags", "merchant risk above training p90.", True),
    # ---- missingness ----------------------------------------------------------------------------
    ("auth_missing", "missing", "authentication_level missing (predictive: 19% vs 11% fraud).", False),
    ("n_missing_numeric", "missing", "number of missing values among the 5 nullable numeric fields.", False),
    ("miss_transaction_amount", "missing", "indicator: amount missing.", False),
    ("miss_distance", "missing", "indicator: distance missing.", False),
    ("miss_merchant_risk", "missing", "indicator: merchant risk missing.", False),
    ("miss_ip_risk", "missing", "indicator: ip risk missing.", False),
    ("miss_login_minutes", "missing", "indicator: login minutes missing.", False),
    # ---- binning --------------------------------------------------------------------------------
    ("amount_bin", "binning", "Quintile (0-4) of amount using training edges.", True),
    ("distance_bin", "binning", "Quintile of distance.", True),
    ("account_age_bin", "binning", "Quintile of account age.", True),
    ("ip_risk_bin", "binning", "Quintile of ip risk.", True),
    ("merchant_risk_bin", "binning", "Quintile of merchant risk.", True),
    ("login_bin", "binning", "Quintile of login minutes.", True),
    # ---- categorical encodings ------------------------------------------------------------------
    ("freq_country", "encoding", "Training share of the customer's country (rarity).", True),
    ("freq_merchant_category", "encoding", "Training share of the merchant category.", True),
    ("freq_device_type", "encoding", "Training share of the device type.", True),
    ("combo_channel_merchant", "encoding", "channel x merchant_category -> smoothed target encoding (cross-fitted inside the pipeline).", True),
    ("combo_auth_channel", "encoding", "authentication x channel -> smoothed target encoding.", True),
    ("combo_country_device", "encoding", "country x device_type -> smoothed target encoding.", True),
    ("combo_tier_channel", "encoding", "customer tier x channel -> smoothed target encoding.", True),
]
ENGINEERED_DOC = {n: d for n, g, d, st in FE_SPEC}
FE_GROUP = {n: g for n, g, d, st in FE_SPEC}
FE_STATEFUL = {n for n, g, d, st in FE_SPEC if st}
GROUPS = ["transform", "velocity", "amount_norm", "risk", "time", "behaviour", "interaction", "flags", "missing", "binning", "encoding"]
TE_COLS = ["combo_channel_merchant", "combo_auth_channel", "combo_country_device", "combo_tier_channel"]
NULLABLE_NUM = ["transaction_amount", "distance_from_usual_location_km", "merchant_risk_score", "ip_risk_score", "login_to_transaction_minutes"]


# --------------------------------------------------------------------------------------
# Feature engineering (stateful -> sklearn transformer, fitted inside the CV/training pipeline)
# --------------------------------------------------------------------------------------
def _grp_stats(x, key):
    g = x.groupby(key)
    m, sd = g.mean().to_dict(), g.std().fillna(1.0).replace(0, 1.0).to_dict()
    return m, sd


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """84 engineered features in 11 groups. All statistics (group means, percentiles, bin edges, frequencies)
    are learned in fit() on TRAINING data only. `groups` lets us switch groups on/off for ablation studies."""

    def __init__(self, groups=None):
        self.groups = groups

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()
        amt, dist = X["transaction_amount"], X["distance_from_usual_location_km"]
        la, ld = np.log1p(amt), np.log1p(dist)
        self.global_amt_median_ = float(amt.median())
        self.tier_amt_median_ = amt.groupby(X["customer_tier"]).median().to_dict()
        self.cat_la_mean_, self.cat_la_std_ = _grp_stats(la, X["merchant_category"])
        self.chan_la_mean_, self.chan_la_std_ = _grp_stats(la, X["channel"])
        self.ctry_ld_mean_, self.ctry_ld_std_ = _grp_stats(ld, X["country"])
        self.global_la_mean_, self.global_la_std_ = float(la.mean()), float(la.std())
        self.global_ld_mean_, self.global_ld_std_ = float(ld.mean()), float(ld.std())
        t24 = X["transactions_last_24h"]
        self.tier_t24_median_ = t24.groupby(X["customer_tier"]).median().to_dict()
        self.chan_t24_median_ = t24.groupby(X["channel"]).median().to_dict()
        self.global_t24_median_ = float(t24.median())
        ll = np.log1p(X["login_to_transaction_minutes"])
        self.login_mean_, self.login_std_ = float(ll.mean()), float(ll.std())
        self.login_p10_ = float(X["login_to_transaction_minutes"].quantile(0.10))
        q = lambda s, p: float(s.quantile(p))
        self.q_ = dict(amt90=q(amt, .9), amt95=q(amt, .95), amt75=q(amt, .75), dist75=q(dist, .75), dist90=q(dist, .9),
                       ip75=q(X["ip_risk_score"], .75), ip90=q(X["ip_risk_score"], .9), mr75=q(X["merchant_risk_score"], .75), mr90=q(X["merchant_risk_score"], .9))
        self.ecdf_amt_, self.ecdf_dist_ = np.sort(amt.dropna().values), np.sort(dist.dropna().values)
        self.edges_ = {c: np.unique(X[c].quantile([.2, .4, .6, .8]).values) for c in
                       ["transaction_amount", "distance_from_usual_location_km", "account_age_days", "ip_risk_score", "merchant_risk_score", "login_to_transaction_minutes"]}
        self.freq_ = {c: X[c].value_counts(normalize=True).to_dict() for c in ["country", "merchant_category", "device_type"]}
        return self

    def transform(self, X: pd.DataFrame):
        X = pd.DataFrame(X).copy(); Q = self.q_
        for c in CAT_RAW:
            X[c] = X[c].astype(object).where(X[c].notna(), "Missing")
        amt, dist = X["transaction_amount"], X["distance_from_usual_location_km"]
        t1, t24, fl = X["transactions_last_1h"], X["transactions_last_24h"], X["failed_logins_24h"]
        nd, mm, pcb = X["new_device"], X["billing_address_mismatch"], X["prior_chargebacks"]
        mr, ip, acc, dev, em = X["merchant_risk_score"], X["ip_risk_score"], X["account_age_days"], X["device_age_days"], X["email_age_days"]
        lg = X["login_to_transaction_minutes"]; hr, dw = X["hour"], X["day_of_week"]
        cnp = (X["card_presence"] == "Not present").astype(int); weak = (X["authentication_level"] != "Strong")
        la, ld = np.log1p(amt), np.log1p(dist)
        gt = lambda s, v: (s > v).astype(float).where(s.notna(), np.nan)
        F = pd.DataFrame(index=X.index)
        # transform
        F["log_amount"], F["log_distance_km"] = la, ld
        F["log_account_age"], F["log_device_age"], F["log_email_age"], F["log_login_minutes"] = np.log1p(acc), np.log1p(dev), np.log1p(em), np.log1p(lg)
        F["sqrt_tx24h"] = np.sqrt(t24)
        F["amount_pct_rank"] = np.searchsorted(self.ecdf_amt_, amt.values, side="right") / len(self.ecdf_amt_); F.loc[amt.isna(), "amount_pct_rank"] = np.nan
        F["distance_pct_rank"] = np.searchsorted(self.ecdf_dist_, dist.values, side="right") / len(self.ecdf_dist_); F.loc[dist.isna(), "distance_pct_rank"] = np.nan
        # velocity
        F["velocity_ratio_1h_24h"] = t1 / (t24 + 1); F["tx_prior_23h"] = t24 - t1
        F["hourly_rate_vs_daily"] = t1 / ((t24 + 1) / 24); F["burst_flag"] = (t1 >= 4).astype(int)
        F["failed_logins_per_tx24"] = fl / (t24 + 1); F["amount_x_tx1h"] = amt * t1
        # amount normalisation
        mu = X["merchant_category"].map(self.cat_la_mean_).fillna(self.global_la_mean_); sd = X["merchant_category"].map(self.cat_la_std_).fillna(self.global_la_std_)
        F["amount_z_by_merchant_cat"] = (la - mu) / sd
        mu = X["channel"].map(self.chan_la_mean_).fillna(self.global_la_mean_); sd = X["channel"].map(self.chan_la_std_).fillna(self.global_la_std_)
        F["amount_z_by_channel"] = (la - mu) / sd
        F["amount_to_tier_median"] = amt / X["customer_tier"].map(self.tier_amt_median_).fillna(self.global_amt_median_)
        F["amount_per_account_day"] = amt / (acc + 1); F["amount_to_device_age"] = amt / (dev + 1)
        F["high_amount_p90"], F["high_amount_p95"] = gt(amt, Q["amt90"]), gt(amt, Q["amt95"])
        # risk
        F["combined_risk_mean"] = X[["merchant_risk_score", "ip_risk_score"]].mean(axis=1, skipna=False)
        F["combined_risk_max"] = X[["merchant_risk_score", "ip_risk_score"]].max(axis=1, skipna=False)
        F["risk_product"], F["risk_gap"] = mr * ip, (mr - ip).abs()
        F["ip_risk_sq"], F["merchant_risk_sq"] = ip ** 2, mr ** 2
        F["both_risk_high"] = ((mr > Q["mr75"]) & (ip > Q["ip75"])).astype(int)
        flags = [nd, mm, (pcb > 0), (fl >= 1), (ip > Q["ip90"]), (mr > Q["mr90"]), (lg < self.login_p10_), (acc < 90), (weak & (cnp == 1)), (dist > Q["dist90"]), (amt > Q["amt90"])]
        F["red_flag_count"] = sum(f.fillna(False).astype(int) if hasattr(f, "fillna") else f for f in flags)
        # time
        F["is_night"] = (hr <= 5).astype(int); F["is_weekend"] = (dw >= 5).astype(int)
        F["is_business_hours"] = ((dw < 5) & (hr >= 9) & (hr <= 17)).astype(int)
        F["hour_sin"], F["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
        F["dow_sin"], F["dow_cos"] = np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)
        F["time_block"] = np.select([hr <= 5, hr <= 11, hr <= 17], [0, 1, 2], 3); F["night_x_cnp"] = F["is_night"] * cnp
        # behavioural deviation
        F["tx24h_dev_from_tier_median"] = t24 - X["customer_tier"].map(self.tier_t24_median_).fillna(self.global_t24_median_)
        F["tx24h_dev_from_channel_median"] = t24 - X["channel"].map(self.chan_t24_median_).fillna(self.global_t24_median_)
        mu = X["country"].map(self.ctry_ld_mean_).fillna(self.global_ld_mean_); sd = X["country"].map(self.ctry_ld_std_).fillna(self.global_ld_std_)
        F["distance_z_by_country"] = (ld - mu) / sd
        F["login_z"] = (np.log1p(lg) - self.login_mean_) / self.login_std_
        F["fast_login_flag"] = (lg < self.login_p10_).astype(float).where(lg.notna(), np.nan)
        F["device_to_account_age_ratio"], F["email_to_account_age_ratio"] = dev / (acc + 1), em / (acc + 1)
        F["device_older_than_account"] = (dev > acc).astype(int); F["new_device_inconsistent"] = ((nd == 1) & (dev > 30)).astype(int)
        # interactions
        F["distance_x_new_device"] = ld * nd; F["amount_x_merchant_risk"] = amt * mr; F["amount_x_ip_risk"] = amt * ip
        F["failed_logins_x_new_device"] = fl * nd; F["velocity_x_new_device"] = t1 * nd; F["mismatch_x_new_device"] = mm * nd
        F["chargeback_x_new_device"] = pcb * nd; F["weak_auth_cnp"] = (weak & (cnp == 1)).astype(int)
        F["new_device_x_cnp"] = nd * cnp; F["mismatch_x_cnp"] = mm * cnp
        F["far_x_high_amount"] = ((dist > Q["dist75"]) & (amt > Q["amt75"])).astype(int)
        # flags
        F["new_account_flag"], F["very_new_account"] = (acc < 90).astype(int), (acc < 30).astype(int)
        F["prior_chargeback_flag"], F["multi_chargeback"] = (pcb > 0).astype(int), (pcb >= 2).astype(int)
        F["failed_login_flag"], F["many_failed_logins"] = (fl >= 1).astype(int), (fl >= 2).astype(int)
        F["young_device"] = (dev < 30).astype(int)
        F["far_distance_flag"], F["high_ip_flag"], F["high_merchant_flag"] = gt(dist, Q["dist90"]), gt(ip, Q["ip90"]), gt(mr, Q["mr90"])
        # missingness
        F["auth_missing"] = (X["authentication_level"] == "Missing").astype(int)
        F["n_missing_numeric"] = X[NULLABLE_NUM].isna().sum(axis=1)
        for nm, c in [("miss_transaction_amount", "transaction_amount"), ("miss_distance", "distance_from_usual_location_km"), ("miss_merchant_risk", "merchant_risk_score"),
                      ("miss_ip_risk", "ip_risk_score"), ("miss_login_minutes", "login_to_transaction_minutes")]:
            F[nm] = X[c].isna().astype(int)
        # binning (NaN stays NaN)
        for nm, c in [("amount_bin", "transaction_amount"), ("distance_bin", "distance_from_usual_location_km"), ("account_age_bin", "account_age_days"),
                      ("ip_risk_bin", "ip_risk_score"), ("merchant_risk_bin", "merchant_risk_score"), ("login_bin", "login_to_transaction_minutes")]:
            b = np.digitize(X[c].values, self.edges_[c]).astype(float); b[X[c].isna().values] = np.nan; F[nm] = b
        # encodings
        for nm, c in [("freq_country", "country"), ("freq_merchant_category", "merchant_category"), ("freq_device_type", "device_type")]:
            F[nm] = X[c].map(self.freq_[c]).fillna(0.0)
        F["combo_channel_merchant"] = X["channel"] + "|" + X["merchant_category"]
        F["combo_auth_channel"] = X["authentication_level"] + "|" + X["channel"]
        F["combo_country_device"] = X["country"] + "|" + X["device_type"]
        F["combo_tier_channel"] = X["customer_tier"] + "|" + X["channel"]
        keep = [n for n in F.columns if self.groups is None or FE_GROUP[n] in self.groups]
        return pd.concat([X, F[keep]], axis=1)


class RawCategoricalFiller(BaseEstimator, TransformerMixin):
    """Used by the 'raw features only' pipelines: only fills categorical NaN with an explicit level."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in CAT_RAW:
            X[c] = X[c].astype(object).where(X[c].notna(), "Missing")
        return X


def engineered_columns(groups=None):
    return [n for n, g, d, st in FE_SPEC if (groups is None or g in groups)]


def _numeric_columns(df):
    """Callable column selector (module-level so that pipelines stay picklable): everything that is not a categorical."""
    return [c for c in df.columns if c not in CAT_RAW and c not in TE_COLS]


def _te_columns(df):
    return [c for c in TE_COLS if c in df.columns]


def make_preprocessor(use_fe: bool = True, scale: bool = False, groups=None) -> Pipeline:
    """[FeatureEngineer] -> impute / scale numerics, one-hot base categoricals, target-encode category combinations."""
    from sklearn.preprocessing import TargetEncoder
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=not use_fe, keep_empty_features=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    parts = [("num", Pipeline(num_steps), _numeric_columns),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_RAW)]
    if use_fe:
        te_steps = [("te", TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=SEED))]
        if scale:
            te_steps.append(("sc", StandardScaler()))
        parts.append(("te", Pipeline(te_steps), _te_columns))
    ct = ColumnTransformer(parts, verbose_feature_names_out=False)
    ct.set_output(transform="default")
    head = ("fe", FeatureEngineer(groups=groups)) if use_fe else ("fill", RawCategoricalFiller())
    return Pipeline([head, ("ct", ct)])


def make_pipeline(model, use_fe=True, scale=False, sampler=None, groups=None, select_k=None):
    """Full pipeline. If a sampler is given an imblearn Pipeline is used so resampling happens only
    on the training part of each fold (never on validation/test data). `select_k` adds in-pipeline
    mutual-information feature selection (fitted on training folds only)."""
    prep = make_preprocessor(use_fe, scale, groups)
    steps = list(prep.steps)
    if select_k:
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        from functools import partial
        steps.append(("select", SelectKBest(partial(mutual_info_classif, random_state=SEED), k=select_k)))
    if sampler is None:
        return Pipeline([("prep", Pipeline(steps)), ("model", model)])
    from imblearn.pipeline import Pipeline as ImbPipeline
    # imblearn forbids a nested Pipeline as an intermediate step -> flatten the preprocessing steps
    return ImbPipeline(steps + [("sampler", sampler), ("model", model)])


def feature_names(pipe) -> list:
    prep = pipe.named_steps["prep"]
    names = list(prep.named_steps["ct"].get_feature_names_out())
    if "select" in prep.named_steps:
        names = [n for n, k in zip(names, prep.named_steps["select"].get_support()) if k]
    return names


# --------------------------------------------------------------------------------------
# Business cost framework
# --------------------------------------------------------------------------------------
@dataclass
class CostParams:
    """Hypothetical per-transaction costs in USD. All values are ASSUMPTIONS (see README)."""
    fn_fee: float = 15.0            # fixed chargeback / dispute handling fee added to a missed fraud's amount
    fp_block_fixed: float = 40.0    # cost of wrongly blocking a good customer (support call, churn risk)
    fp_block_margin: float = 0.02   # lost merchant/interchange margin as share of the blocked amount
    review_cost: float = 20.0       # analyst time per manually reviewed transaction
    review_friction: float = 3.0    # delay friction when a legitimate transaction is reviewed
    analyst_recall: float = 0.90    # share of reviewed frauds the analyst correctly stops

    def as_dict(self):
        return asdict(self)


def policy_costs(y, amount, decision, p: CostParams):
    """Per-transaction cost vector. decision: 0 approve, 1 review, 2 block."""
    y = np.asarray(y); amount = np.asarray(amount, float); d = np.asarray(decision)
    loss = amount + p.fn_fee
    c_app = y * loss
    c_rev = p.review_cost + y * (1 - p.analyst_recall) * loss + (1 - y) * p.review_friction
    c_blk = (1 - y) * (p.fp_block_fixed + p.fp_block_margin * amount)
    return np.where(d == 0, c_app, np.where(d == 1, c_rev, c_blk))


def decide(prob, t_review, t_block):
    prob = np.asarray(prob)
    return np.where(prob >= t_block, 2, np.where(prob >= t_review, 1, 0))


def policy_total_cost(y, amount, prob, t_review, t_block, p: CostParams):
    return policy_costs(y, amount, decide(prob, t_review, t_block), p).sum()


def optimise_thresholds(y, amount, prob, p: CostParams, grid=None, max_review_share=1.0, min_review_share=0.0, min_block_share=0.0):
    """Exhaustive search of (t_review <= t_block) minimising total expected cost.
    Returns (t_review, t_block, total_cost). t_review == t_block means 'no review band';
    t_block = 1.01 means 'never block'. `min_review_share` / `min_block_share` force a genuine three-way policy
    (each band must receive at least that share of volume); `max_review_share` models analyst capacity."""
    y = np.asarray(y); amount = np.asarray(amount, float); prob = np.asarray(prob)
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2) if grid is None else grid
    ts = np.concatenate([grid, [1.01]])
    best = (None, None, np.inf)
    # precompute per-row costs of each action once
    loss = amount + p.fn_fee
    c0 = y * loss
    c1 = p.review_cost + y * (1 - p.analyst_recall) * loss + (1 - y) * p.review_friction
    c2 = (1 - y) * (p.fp_block_fixed + p.fp_block_margin * amount)
    order = np.argsort(prob)
    prob_s, c0s, c1s, c2s = prob[order], c0[order], c1[order], c2[order]
    cs0 = np.concatenate([[0], np.cumsum(c0s)]); cs1 = np.concatenate([[0], np.cumsum(c1s)])
    cs2 = np.concatenate([[0], np.cumsum(c2s)])
    idx = {t: np.searchsorted(prob_s, t, side="left") for t in ts}
    n = len(prob)
    for i, tr in enumerate(ts):
        a = idx[tr]
        for tb in ts[i:]:
            b = idx[tb]
            if (b - a) / n > max_review_share or (b - a) / n < min_review_share or (n - b) / n < min_block_share:
                continue
            tot = cs0[a] + (cs1[b] - cs1[a]) + (cs2[n] - cs2[b])
            if tot < best[2]:
                best = (float(tr), float(tb), float(tot))
    return best


def single_threshold_cost(y, amount, prob, t, p: CostParams):
    """Binary policy: block if prob >= t else approve (no manual review)."""
    return policy_costs(y, amount, np.where(np.asarray(prob) >= t, 2, 0), p).sum()


def bayes_decision(prob, amount, p: CostParams):
    """Amount-aware expected-cost-minimising action (needs calibrated probabilities)."""
    prob = np.asarray(prob); amount = np.asarray(amount, float)
    loss = amount + p.fn_fee
    e0 = prob * loss
    e1 = p.review_cost + prob * (1 - p.analyst_recall) * loss + (1 - prob) * p.review_friction
    e2 = (1 - prob) * (p.fp_block_fixed + p.fp_block_margin * amount)
    return np.argmin(np.vstack([e0, e1, e2]), axis=0)


# --------------------------------------------------------------------------------------
# Metrics / monitoring helpers
# --------------------------------------------------------------------------------------
def expected_calibration_error(y, prob, bins=10):
    y = np.asarray(y); prob = np.asarray(prob)
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1)); edges[-1] += 1e-9
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi)
        if m.sum():
            ece += m.mean() * abs(y[m].mean() - prob[m].mean())
    return ece


def psi(expected, actual, bins=10):
    """Population Stability Index between a reference and a new sample (numeric)."""
    e = pd.Series(expected).dropna().values; a = pd.Series(actual).dropna().values
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e); pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))
