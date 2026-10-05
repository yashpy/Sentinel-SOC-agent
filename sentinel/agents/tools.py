"""Read-only investigation tools over the audit log.

Tools return compact, numbered *facts* that compare the day to the user's own history
(context engineering): the LLM never sees thousands of raw log lines. Ground-truth columns are
dropped on load so neither tools nor agents can see labels.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from sentinel import config
from sentinel.ml.features import HIGH_RISK_PERMS, RISKY_POLICIES, add_local_day

INVESTIGATORS = ("auth", "data_access", "privilege", "session")
RAW_COLUMNS = ["ts", "event_type", "ip", "country", "asn_type", "device_id", "session_id", "status", "records",
               "detail", "target_user"]


class EventStore:
    def __init__(self) -> None:
        events = pd.read_parquet(config.EVENTS_PATH).drop(columns=config.LABEL_COLUMNS)
        self.users = pd.read_parquet(config.USERS_PATH).set_index("user_id")
        self.ev = add_local_day(events, self.users.reset_index())
        self.by_user = {u: g.sort_values("ts") for u, g in self.ev.groupby("user_id")}

    def day(self, user: str, day: int) -> pd.DataFrame:
        g = self.by_user[user]
        return g[g.day == day]

    def history(self, user: str, day: int, lookback: int = 30) -> pd.DataFrame:
        g = self.by_user[user]
        return g[(g.day < day) & (g.day >= day - lookback)]

    def raw_log(self, user: str, day: int, max_rows: int = 120) -> tuple[str, int]:
        d = self.day(user, day)[RAW_COLUMNS]
        return d.head(max_rows).to_csv(index=False), len(d)


@lru_cache(maxsize=1)
def get_store() -> EventStore:
    return EventStore()


def _fmt_ip(row_ip: str, asn: str, cc: str) -> str:
    return f"{row_ip} ({asn}, {cc})"


def user_profile(store: EventStore, user: str, day: int) -> dict:
    u = store.users.loc[user]
    h = store.history(user, day)
    logins = h[(h.event_type == "login") & (h.status == "success")]
    return dict(user_id=user, role=u.role, home_country=u.country, is_admin=bool(u.is_admin),
                typical_login_hour=round(float(logins.local_hour.median()), 1) if len(logins) else None,
                known_devices=int(h.device_id[h.device_id != "api-client"].nunique()),
                known_ips=int(h.ip.nunique()), active_days_last_30=int(h.day.nunique()))


def investigate_auth(store: EventStore, user: str, day: int) -> list[str]:
    d, h = store.day(user, day), store.history(user, day)
    known_ips, known_dev, known_cc = set(h.ip), set(h.device_id), set(h.country) | {store.users.at[user, "country"]}
    facts = []
    fails = d[(d.event_type == "login") & (d.status == "failure")]
    oks = d[(d.event_type == "login") & (d.status == "success")]
    hist_fail_days = h[(h.event_type == "login") & (h.status == "failure")].groupby("day").size()
    facts.append(f"{len(fails)} failed and {len(oks)} successful logins today; the user averaged "
                 f"{hist_fail_days.sum() / max(h.day.nunique(), 1):.2f} failed logins/day over the last 30 days.")
    for ip, g in fails.groupby("ip"):
        later_ok = ((oks.ip == ip) & (oks.ts > g.ts.min())).any()
        facts.append(f"{len(g)} failed logins from {_fmt_ip(ip, g.asn_type.iloc[0], g.country.iloc[0])} "
                     f"over {(g.ts.max() - g.ts.min()).total_seconds() / 60:.0f} min; "
                     f"{'IP previously used by this user' if ip in known_ips else 'IP never seen for this user'}; "
                     f"{'followed by a SUCCESSFUL login from the same IP' if later_ok else 'no successful login from it'}.")
    for _, r in oks.iterrows():
        tags = [("new IP" if r.ip not in known_ips else "known IP"),
                ("new device" if r.device_id not in known_dev else "known device"),
                ("new country" if r.country not in known_cc else "known country")]
        facts.append(f"Successful login at local hour {r.local_hour:.1f} from {_fmt_ip(r.ip, r.asn_type, r.country)}: "
                     f"{', '.join(tags)}.")
    mfa = d[d.event_type == "mfa_challenge"]
    denied = mfa[mfa.status == "denied"]
    if len(denied):
        facts.append(f"{len(denied)} MFA push challenges DENIED from {denied.ip.nunique()} IP(s) "
                     f"({', '.join(sorted(set(denied.country)))}) within "
                     f"{(denied.ts.max() - denied.ts.min()).total_seconds() / 60:.0f} min; "
                     f"{(mfa.status == 'approved').sum()} approved.")
    return facts[:12]


def investigate_data_access(store: EventStore, user: str, day: int) -> list[str]:
    d, h = store.day(user, day), store.history(user, day)
    days = max(h.day.nunique(), 1)
    facts = []
    exp, hexp = d[d.event_type == "export"], h[h.event_type == "export"]
    base_day = hexp.records.sum() / days
    base_single = float(hexp.records.median()) if len(hexp) else 0.0
    facts.append(f"{len(exp)} exports totalling {int(exp.records.sum()):,} records today vs a baseline of "
                 f"{base_day:,.0f} exported records/day (median single export {base_single:,.0f}); "
                 f"ratio {exp.records.sum() / max(base_day, 1):.1f}x.")
    if len(exp):
        ips = ", ".join(sorted({_fmt_ip(i, a, c) for i, a, c in zip(exp.ip, exp.asn_type, exp.country)}))[:200]
        facts.append(f"Largest single export {int(exp.records.max()):,} records; exports came from {ips}; "
                     f"local hours {exp.local_hour.min():.1f}-{exp.local_hour.max():.1f}.")
    acc, hacc = d[d.event_type == "record_access"], h[h.event_type == "record_access"]
    facts.append(f"{int(acc.records.sum()):,} records viewed today vs baseline {hacc.records.sum() / days:,.0f}/day "
                 f"({acc.records.sum() / max(hacc.records.sum() / days, 1):.1f}x).")
    api, hapi = d[d.event_type == "api_query"], h[h.event_type == "api_query"]
    if len(api) or len(hapi):
        src = api.groupby(["ip", "asn_type", "country"]).size().sort_values(ascending=False).head(3)
        src_s = "; ".join(f"{n} from {_fmt_ip(i, a, c)}{' (new IP)' if i not in set(h.ip) else ''}"
                          for (i, a, c), n in src.items())
        facts.append(f"{len(api)} API queries ({int(api.records.sum()):,} records) today vs baseline "
                     f"{len(hapi) / days:.1f} queries/day. Sources: {src_s or 'none'}.")
    tok = d[d.event_type == "token_create"]
    if len(tok):
        facts.append(f"{len(tok)} new API access token(s) created at local hour {tok.local_hour.min():.1f} from "
                     f"{_fmt_ip(tok.ip.iloc[0], tok.asn_type.iloc[0], tok.country.iloc[0])}; user created "
                     f"{(h.event_type == 'token_create').sum()} tokens in the previous 30 days.")
    return facts


def investigate_privilege(store: EventStore, user: str, day: int) -> list[str]:
    d, h = store.day(user, day), store.history(user, day)
    is_admin = bool(store.users.at[user, "is_admin"])
    facts = [f"User role is {store.users.at[user, 'role']} ({'admin' if is_admin else 'NOT an admin'}); "
             f"made {(h.event_type == 'permission_grant').sum()} permission grants in the previous 30 days."]
    g = d[d.event_type == "permission_grant"]
    for _, r in g.head(8).iterrows():
        facts.append(f"Granted permission {r.detail} ({'HIGH-RISK' if r.detail in HIGH_RISK_PERMS else 'standard'}) "
                     f"to {'THEMSELVES' if r.target_user == user else r.target_user} at local hour {r.local_hour:.1f} "
                     f"from {_fmt_ip(r.ip, r.asn_type, r.country)}.")
    if len(g) > 8:
        facts.append(f"... plus {len(g) - 8} more grants ({g.detail.isin(HIGH_RISK_PERMS).sum()} high-risk in total).")
    for _, r in d[d.event_type == "auth_policy_change"].iterrows():
        facts.append(f"Changed authentication policy: {r.detail} "
                     f"({'WEAKENS security' if r.detail in RISKY_POLICIES else 'routine change'}) from "
                     f"{_fmt_ip(r.ip, r.asn_type, r.country)}, device {'known' if r.device_id in set(h.device_id) else 'NEW'}.")
    if len(facts) == 1:
        facts.append("No permission grants or authentication-policy changes today.")
    return facts


def investigate_session(store: EventStore, user: str, day: int) -> list[str]:
    d = store.day(user, day).sort_values("ts")
    full = store.by_user[user]
    login_ip = full[(full.event_type == "login") & (full.status == "success")].set_index("session_id")["ip"]
    facts = []
    s = d[d.session_id != ""]
    mism = s[s.session_id.map(login_ip).notna() & (s.ip != s.session_id.map(login_ip))]
    if len(mism):
        for sid, g in mism.groupby("session_id"):
            facts.append(f"Session {sid} was created from {login_ip[sid]} but {len(g)} events reused it from "
                         f"{_fmt_ip(g.ip.iloc[0], g.asn_type.iloc[0], g.country.iloc[0])} with no new login.")
    else:
        facts.append("All session activity came from the IP that created the session.")
    cc = d.country.values
    if len(d) > 1:
        idx = np.where((cc[1:] != cc[:-1]))[0]
        for i in idx[:2]:
            gap = (d.ts.iloc[i + 1] - d.ts.iloc[i]).total_seconds() / 60
            if gap < 120:
                facts.append(f"Impossible travel: activity in {cc[i]} then {cc[i + 1]} only {gap:.0f} minutes apart.")
    hosting = (d.asn_type == "hosting").mean()
    facts.append(f"{hosting:.0%} of today's events came from hosting-provider/datacenter IPs.")
    return facts


TOOLS = {
    "auth": investigate_auth,
    "data_access": investigate_data_access,
    "privilege": investigate_privilege,
    "session": investigate_session,
}
