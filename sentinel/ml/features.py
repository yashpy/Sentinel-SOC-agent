"""User-day behavioural features, computed against each user's *prior* history only (no leakage).

Usage: python -m sentinel.ml.features
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from sentinel import config
from sentinel.data.generator import START
from sentinel.data.scenarios import ATTACKS

HIGH_RISK_PERMS = {"Modify_All_Data", "View_All_Data", "Manage_Users", "Export_Reports", "API_Enabled"}
RISKY_POLICIES = {"mfa_disabled", "trusted_ip_range_added", "password_policy_relaxed"}

FEATURE_COLUMNS = [
    "n_events", "n_login_ok", "n_login_fail", "max_fails_one_ip", "fail_then_success", "n_mfa_denied",
    "n_ips", "n_new_ips", "n_new_countries", "n_new_devices", "foreign_country", "frac_hosting",
    "login_hour_dev", "frac_offhours", "n_exports", "export_records_log", "export_ratio", "max_export_ratio",
    "access_ratio", "n_api", "api_ratio", "api_hosting", "n_token_create", "n_grants", "n_highrisk_grants",
    "n_self_grants", "nonadmin_grant", "n_policy_changes", "n_risky_policy", "session_ip_mismatch",
    "impossible_travel", "is_admin", "is_engineer", "weekend",
]


def add_local_day(events: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    tz = events["user_id"].map(users.set_index("user_id")["tz_offset"])
    local = events["ts"] + pd.to_timedelta(tz, unit="h")
    out = events.copy()
    out["local_hour"] = local.dt.hour + local.dt.minute / 60
    out["day"] = (local - pd.Timestamp(START)).dt.days
    return out


def _circ_dev(h: np.ndarray, center: float) -> np.ndarray:
    d = np.abs(h - center) % 24
    return np.minimum(d, 24 - d)


def build_features(events: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    ev = add_local_day(events, users)
    login_ip = ev.loc[(ev.event_type == "login") & (ev.status == "success")].set_index("session_id")["ip"].to_dict()
    ev["sess_login_ip"] = ev["session_id"].map(login_ip)
    uinfo = users.set_index("user_id")
    rows = []
    for uid, g in ev.groupby("user_id", sort=False):
        home, is_admin, role = uinfo.at[uid, "country"], bool(uinfo.at[uid, "is_admin"]), uinfo.at[uid, "role"]
        seen_ip, seen_cc, seen_dev = set(), {home}, set()
        hist = defaultdict(list)  # rolling history of daily aggregates
        for day, d in g.groupby("day", sort=True):
            et, st = d.event_type.values, d.status.values
            login_ok = (et == "login") & (st == "success")
            login_fail = (et == "login") & (st == "failure")
            exports = d.records.values[et == "export"]
            api = et == "api_query"
            grants = d[et == "permission_grant"]
            pol = d.detail.values[et == "auth_policy_change"]
            ips, ccs = set(d.ip), set(d.country)
            devs = set(d.device_id) - {"api-client"}
            fails_by_ip = pd.Series(d.ip.values[login_fail]).value_counts()
            fail_ips = set(fails_by_ip.index)
            ok_ips = set(d.ip.values[login_ok])
            login_hours = d.local_hour.values[login_ok]
            med_hour = float(np.median(hist["login_hour"])) if hist["login_hour"] else 9.0
            base_exp = np.mean(hist["export_day"]) if hist["export_day"] else 0.0
            base_single = np.median(hist["export_single"]) if hist["export_single"] else 150.0
            base_acc = np.mean(hist["access_day"]) if hist["access_day"] else 100.0
            base_api = np.mean(hist["api_day"]) if hist["api_day"] else 0.0
            access_rec = d.records.values[et == "record_access"].sum()
            api_rec = d.records.values[api].sum()
            sess = d[d.session_id != ""]
            mismatch = int(((sess.sess_login_ip.notna()) & (sess.ip != sess.sess_login_ip)).sum())
            # impossible travel: consecutive events in different countries within 2 hours
            dd = d.sort_values("ts")
            cc_change = (dd.country.values[1:] != dd.country.values[:-1])
            gap_h = np.diff(dd.ts.values).astype("timedelta64[s]").astype(float) / 3600
            travel = int(np.any(cc_change & (gap_h < 2))) if len(dd) > 1 else 0
            exp_sum = float(exports.sum())
            rows.append(dict(
                user_id=uid, day=int(day),
                n_events=len(d), n_login_ok=int(login_ok.sum()), n_login_fail=int(login_fail.sum()),
                max_fails_one_ip=int(fails_by_ip.max()) if len(fails_by_ip) else 0,
                fail_then_success=int(len(fail_ips & ok_ips) > 0 and login_fail.sum() >= 3),
                n_mfa_denied=int(((et == "mfa_challenge") & (st == "denied")).sum()),
                n_ips=len(ips), n_new_ips=len(ips - seen_ip) if seen_ip else 0,
                n_new_countries=len(ccs - seen_cc), n_new_devices=len(devs - seen_dev) if seen_dev else 0,
                foreign_country=int(any(c != home for c in ccs)),
                frac_hosting=float((d.asn_type == "hosting").mean()),
                login_hour_dev=float(_circ_dev(login_hours, med_hour).max()) if len(login_hours) else 0.0,
                frac_offhours=float(((d.local_hour < 6) | (d.local_hour > 21)).mean()),
                n_exports=len(exports), export_records_log=float(np.log1p(exp_sum)),
                export_ratio=float(np.log1p(exp_sum) - np.log1p(base_exp)),
                max_export_ratio=float(np.log1p(exports.max()) - np.log1p(base_single)) if len(exports) else 0.0,
                access_ratio=float(np.log1p(access_rec) - np.log1p(base_acc)),
                n_api=int(api.sum()), api_ratio=float(np.log1p(api_rec) - np.log1p(base_api)),
                api_hosting=int((api & (d.asn_type.values == "hosting")).sum()),
                n_token_create=int((et == "token_create").sum()),
                n_grants=len(grants), n_highrisk_grants=int(grants.detail.isin(HIGH_RISK_PERMS).sum()),
                n_self_grants=int((grants.target_user == uid).sum()),
                nonadmin_grant=int((not is_admin) and len(grants) > 0),
                n_policy_changes=len(pol), n_risky_policy=int(np.isin(pol, list(RISKY_POLICIES)).sum()),
                session_ip_mismatch=mismatch, impossible_travel=travel,
                is_admin=int(is_admin), is_engineer=int(role == "engineer"),
                weekend=int((pd.Timestamp(START) + pd.Timedelta(days=int(day))).weekday() >= 5),
                # ---- labels (never used as model inputs) ----
                label=int(d.scenario.isin(list(ATTACKS)).any()),
                scenario=_day_scenario(d),
                incident_id=",".join(sorted(set(d.incident_id) - {""})),
            ))
            # update history AFTER computing features for the day
            seen_ip |= ips
            seen_cc |= ccs
            seen_dev |= devs
            hist["login_hour"].extend(login_hours.tolist())
            hist["export_day"].append(exp_sum)
            hist["export_single"].extend(exports.tolist())
            hist["access_day"].append(access_rec)
            hist["api_day"].append(api_rec)
    return pd.DataFrame(rows).sort_values(["day", "user_id"]).reset_index(drop=True)


def _day_scenario(d: pd.DataFrame) -> str:
    scen = set(d.scenario) - {""}
    mal = [s for s in scen if s in ATTACKS]
    if mal:
        return mal[0]
    return next(iter(scen)) if scen else "normal"


def main() -> None:
    events = pd.read_parquet(config.EVENTS_PATH)
    users = pd.read_parquet(config.USERS_PATH)
    feats = build_features(events, users)
    feats.to_parquet(config.FEATURES_PATH, index=False)
    print(f"user-days={len(feats):,} malicious={int(feats.label.sum())} "
          f"benign-lookalike={int(((feats.label == 0) & (feats.scenario != 'normal')).sum())}")


if __name__ == "__main__":
    main()
