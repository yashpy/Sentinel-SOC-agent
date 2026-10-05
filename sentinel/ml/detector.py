"""Threat detector: SIEM-rule baseline vs IsolationForest vs XGBoost, with a time-based split.

The operating threshold is chosen on out-of-fold *training* predictions (target recall), then
frozen and applied to the held-out future test period, as in a real deployment.

Usage: python -m sentinel.ml.detector
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupKFold

from sentinel import config
from sentinel.ml.features import FEATURE_COLUMNS

MODEL_PATH = config.ARTIFACTS_DIR / "xgb_detector.json"
META_PATH = config.ARTIFACTS_DIR / "detector_meta.json"
SCORED_PATH = config.GEN_DIR / "scored_test.parquet"
DEV_SCORED_PATH = config.GEN_DIR / "scored_dev.parquet"
TARGET_RECALL = 0.90

XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
                  min_child_weight=2, reg_lambda=1.0, eval_metric="aucpr", random_state=0, n_jobs=8)


def rule_baseline(f: pd.DataFrame) -> pd.Series:
    """Typical hand-written SIEM detections."""
    return (
        (f.n_login_fail >= 10) | (f.n_mfa_denied >= 3) | (f.n_new_countries >= 1)
        | (f.max_export_ratio > np.log(5)) | (f.n_self_grants >= 1) | (f.nonadmin_grant == 1)
        | (f.n_risky_policy >= 1) | ((f.n_token_create >= 1) & (f.api_hosting > 0)) | (f.frac_hosting > 0.5)
        | (f.session_ip_mismatch >= 1)
    ).astype(int)


def critical_rules(f: pd.DataFrame) -> pd.Series:
    """Few, high-severity deterministic detections that always page (never suppressed by the model)."""
    return ((f.n_risky_policy >= 1) | (f.n_self_grants >= 1) | (f.n_mfa_denied >= 5)).astype(int)


def _metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None, f: pd.DataFrame) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    lookalike = (f.scenario != "normal").values & (y == 0)
    n_days = f.day.nunique()
    out = dict(precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
               alerts=int(pred.sum()), alerts_per_day=round(pred.sum() / n_days, 1),
               false_positives=int(((pred == 1) & (y == 0)).sum()),
               lookalike_fp_rate=round(float(pred[lookalike].mean()), 4) if lookalike.any() else None,
               normal_fp_rate=round(float(pred[(f.scenario == "normal").values].mean()), 5))
    if score is not None:
        out.update(pr_auc=round(average_precision_score(y, score), 4), roc_auc=round(roc_auc_score(y, score), 4))
    return out


def per_scenario_recall(f: pd.DataFrame, pred: np.ndarray) -> dict:
    mal = f.label == 1
    return {s: round(float(pred[(mal & (f.scenario == s)).values].mean()), 3)
            for s in sorted(f.loc[mal, "scenario"].unique())}


def train_and_evaluate() -> dict:
    feats = pd.read_parquet(config.FEATURES_PATH)
    feats = feats[feats.day >= config.WARMUP_DAYS].reset_index(drop=True)
    train = feats[feats.day <= config.TRAIN_END_DAY].reset_index(drop=True)
    test = feats[feats.day > config.TRAIN_END_DAY].reset_index(drop=True)
    Xtr, ytr, Xte, yte = train[FEATURE_COLUMNS], train.label.values, test[FEATURE_COLUMNS], test.label.values
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    # Out-of-fold predictions grouped by day -> threshold selection without touching the test set.
    oof = np.zeros(len(train))
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(Xtr, ytr, groups=train.day):
        m = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        m.fit(Xtr.iloc[tr_idx], ytr[tr_idx])
        oof[va_idx] = m.predict_proba(Xtr.iloc[va_idx])[:, 1]
    cand = np.sort(np.unique(oof))[::-1]
    threshold = next(t for t in cand if (oof[ytr == 1] >= t).mean() >= TARGET_RECALL)

    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
    model.fit(Xtr, ytr)
    model.save_model(MODEL_PATH)
    score = model.predict_proba(Xte)[:, 1]
    pred = (score >= threshold).astype(int)

    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(Xtr[ytr == 0])
    iso_tr = -iso.score_samples(Xtr)
    iso_thr = np.quantile(iso_tr[ytr == 1], 1 - TARGET_RECALL)  # same recall target on train
    iso_score = -iso.score_samples(Xte)
    iso_pred = (iso_score >= iso_thr).astype(int)

    rules_pred = rule_baseline(test).values
    hybrid_pred = (pred | critical_rules(test).values).astype(int)

    # SHAP contributions (native XGBoost TreeSHAP) for analyst-facing explanations.
    contribs = model.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]
    scored = test.copy()
    scored["score"] = score
    scored["ml_alert"] = pred
    scored["alert"] = hybrid_pred
    scored["rule_alert"] = rules_pred
    for j, c in enumerate(FEATURE_COLUMNS):
        scored[f"shap_{c}"] = contribs[:, j]
    scored.to_parquet(SCORED_PATH, index=False)

    # Dev queue from the TRAINING period (out-of-fold scores) for agent/prompt development, so the
    # test-period alerts stay untouched until the final evaluation.
    dev = train.copy()
    dev["score"] = oof
    dev["ml_alert"] = (oof >= threshold).astype(int)
    dev["alert"] = (dev.ml_alert | critical_rules(train).values).astype(int)
    dev["rule_alert"] = rule_baseline(train).values
    dev_contribs = model.get_booster().predict(xgb.DMatrix(Xtr), pred_contribs=True)[:, :-1]
    for j, c in enumerate(FEATURE_COLUMNS):
        dev[f"shap_{c}"] = dev_contribs[:, j]
    dev.to_parquet(DEV_SCORED_PATH, index=False)

    gain = model.get_booster().get_score(importance_type="gain")
    importance = dict(sorted(((k, round(v, 2)) for k, v in gain.items()), key=lambda kv: -kv[1])[:15])
    results = dict(
        split=dict(train_days=f"{config.WARMUP_DAYS}-{config.TRAIN_END_DAY}",
                   test_days=f"{config.TRAIN_END_DAY + 1}-{config.N_DAYS - 1}",
                   train_user_days=len(train), test_user_days=len(test),
                   train_malicious=int(ytr.sum()), test_malicious=int(yte.sum()),
                   test_benign_lookalikes=int(((test.scenario != "normal") & (yte == 0)).sum())),
        threshold=round(float(threshold), 5), target_recall=TARGET_RECALL,
        rules=_metrics(yte, rules_pred, None, test),
        isolation_forest=_metrics(yte, iso_pred, iso_score, test),
        xgboost=_metrics(yte, pred, score, test),
        hybrid=_metrics(yte, hybrid_pred, None, test),
        xgboost_recall_by_scenario=per_scenario_recall(test, pred),
        hybrid_recall_by_scenario=per_scenario_recall(test, hybrid_pred),
        rules_recall_by_scenario=per_scenario_recall(test, rules_pred),
        top_features_gain=importance,
    )
    r, x = results["rules"], results["hybrid"]
    results["headline"] = dict(
        alert_reduction_vs_rules=round(1 - x["alerts"] / max(r["alerts"], 1), 4),
        fp_reduction_vs_rules=round(1 - x["false_positives"] / max(r["false_positives"], 1), 4),
        precision_gain_vs_rules=round(x["precision"] - r["precision"], 4),
    )
    META_PATH.write_text(json.dumps(dict(threshold=float(threshold), features=FEATURE_COLUMNS), indent=2))
    (config.RESULTS_DIR / "detector_metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
