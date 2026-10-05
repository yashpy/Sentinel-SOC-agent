"""Turn detector output into label-free alerts for the agent."""
from __future__ import annotations

import pandas as pd

from sentinel import config
from sentinel.ml.detector import DEV_SCORED_PATH, SCORED_PATH
from sentinel.ml.features import FEATURE_COLUMNS

READABLE = {
    "n_login_fail": "failed logins", "max_fails_one_ip": "failed logins from one IP", "n_mfa_denied": "MFA pushes denied",
    "n_new_ips": "new IPs", "n_new_countries": "new countries", "n_new_devices": "new devices",
    "foreign_country": "activity outside home country", "frac_hosting": "share of datacenter-IP events",
    "login_hour_dev": "login-hour deviation (h)", "frac_offhours": "share of off-hours events",
    "export_ratio": "log export volume vs baseline", "max_export_ratio": "log largest export vs baseline",
    "access_ratio": "log record views vs baseline", "api_ratio": "log API volume vs baseline",
    "api_hosting": "API calls from datacenter IPs", "n_token_create": "API tokens created",
    "n_highrisk_grants": "high-risk permission grants", "n_self_grants": "self permission grants",
    "nonadmin_grant": "grant by non-admin", "n_risky_policy": "auth policy weakened",
    "session_ip_mismatch": "session reused from other IP", "impossible_travel": "impossible travel",
}


def alert_queue(source: str = "union", split: str = "test") -> pd.DataFrame:
    """Alerts for a split. 'union' = ML/hybrid detector OR SIEM rules, i.e. a realistic noisy queue.
    split='dev' uses the training period (out-of-fold scores) for prompt/agent development."""
    sc = pd.read_parquet(SCORED_PATH if split == "test" else DEV_SCORED_PATH)
    mask = {"ml": sc.alert == 1, "rules": sc.rule_alert == 1, "union": (sc.alert == 1) | (sc.rule_alert == 1)}[source]
    return sc[mask].reset_index(drop=True)


def to_alert(row: pd.Series, top_k: int = 6) -> dict:
    contrib = sorted(((row[f"shap_{c}"], c) for c in FEATURE_COLUMNS), reverse=True)
    signals = [f"{READABLE.get(c, c)}={row[c]:.2f} (+{v:.2f})" for v, c in contrib[:top_k] if v > 0.05]
    return dict(alert_id=f"ALERT-{row.user_id}-d{int(row.day)}", user_id=row.user_id, day=int(row.day),
                score=round(float(row.score), 4), signals=signals, ml_alert=bool(row.alert),
                source="ml+rules" if row.alert and row.rule_alert else ("ml" if row.alert else "rules"))


def ground_truth(row: pd.Series) -> dict:
    return dict(label=int(row.label), scenario=row.scenario)


__all__ = ["alert_queue", "to_alert", "ground_truth", "config"]
