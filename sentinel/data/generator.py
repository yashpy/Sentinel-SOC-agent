"""Reproducible SaaS audit-log simulator with labeled attack and benign look-alike incidents.

Produces a Salesforce-style event log (logins, MFA, record access, exports, API calls,
permission grants, token creation, auth-policy changes) for a synthetic org. Incidents are
injected with a random "stealth" level so detection is not trivial, and benign look-alikes
(travel, VPN, quarter-end exports, onboarding...) create realistic false-positive pressure.

Usage: python -m sentinel.data.generator --users 400 --seed 7
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sentinel import config
from sentinel.data.scenarios import ATTACKS, BENIGN

START = datetime(2026, 8, 3)  # a Monday

HOME_COUNTRIES = {"US": 0.48, "IN": 0.15, "GB": 0.10, "DE": 0.08, "CA": 0.06, "BR": 0.05, "JP": 0.04, "AU": 0.04}
TZ_OFFSET = {"US": -6, "IN": 5, "GB": 0, "DE": 1, "CA": -5, "BR": -3, "JP": 9, "AU": 10}
FOREIGN_ATTACK_COUNTRIES = ["RU", "CN", "NG", "RO", "VN", "KP", "IR", "UA"]
TRAVEL_COUNTRIES = ["FR", "ES", "IT", "SG", "MX", "AE", "NL", "TH"]
ROLES = {"sales": 0.35, "support": 0.24, "analyst": 0.15, "engineer": 0.18, "admin": 0.08}
NORMAL_PERMS = ["Reports_Viewer", "Sales_Standard", "Support_Agent", "Dashboard_Builder"]
HIGH_RISK_PERMS = ["Modify_All_Data", "View_All_Data", "Manage_Users", "Export_Reports", "API_Enabled"]
RISKY_POLICIES = ["mfa_disabled", "trusted_ip_range_added", "password_policy_relaxed"]
SAFE_POLICIES = ["session_timeout_changed", "login_hours_updated"]
EXPORT_LAMBDA = {"sales": 0.6, "support": 0.2, "analyst": 1.2, "engineer": 0.3, "admin": 0.4}


@dataclass
class User:
    user_id: str
    role: str
    country: str
    home_ip: str
    corp_ip: str
    devices: list[str]
    start_hour: float  # typical local start of day
    export_mu: float  # log-mean records per export
    is_admin: bool
    sessions: list[tuple[str, datetime, str]] = field(default_factory=list)  # (session_id, ts, ip)


class LogSimulator:
    def __init__(self, n_users: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.n_users = n_users
        self.events: list[dict] = []
        self.incidents: list[dict] = []
        self._ip_counter = itertools.count(1)
        self._dev_counter = itertools.count(1)
        self._sess_counter = itertools.count(1)
        self._inc_counter = itertools.count(1)
        self.users = [self._make_user(i) for i in range(n_users)]
        self.corp_ips = {c: self._ip("corporate") for c in HOME_COUNTRIES}

    # ---------- primitives ----------
    def _ip(self, asn: str) -> str:
        n = next(self._ip_counter)
        prefix = {"residential": 73, "corporate": 12, "hosting": 185, "mobile": 100}[asn]
        return f"{prefix}.{(n >> 16) & 255}.{(n >> 8) & 255}.{n & 255}"

    @staticmethod
    def asn_of(ip: str) -> str:
        return {"73": "residential", "12": "corporate", "185": "hosting", "100": "mobile"}[ip.split(".")[0]]

    def _device(self) -> str:
        return f"dev-{next(self._dev_counter):05d}"

    def _session(self) -> str:
        return f"s-{next(self._sess_counter):07d}"

    def _pick(self, weights: dict[str, float]) -> str:
        keys = list(weights)
        p = np.array(list(weights.values()))
        return str(self.rng.choice(keys, p=p / p.sum()))

    def _make_user(self, i: int) -> User:
        role = self._pick(ROLES)
        country = self._pick(HOME_COUNTRIES)
        devices = [self._device() for _ in range(1 + int(self.rng.random() < 0.35))]
        return User(
            user_id=f"u{i:04d}",
            role=role,
            country=country,
            home_ip=self._ip("residential"),
            corp_ip="",  # filled lazily (shared per country)
            devices=devices,
            start_hour=float(np.clip(self.rng.normal(8.8, 1.0), 6, 12)),
            export_mu=float(self.rng.normal(np.log(150), 0.6)),
            is_admin=role == "admin",
        )

    def _utc(self, day: int, local_hour: float, country: str) -> datetime:
        return START + timedelta(days=day, hours=local_hour - TZ_OFFSET.get(country, 0))

    def emit(self, ts: datetime, user: User, event_type: str, ip: str, country: str, device: str,
             session_id: str = "", status: str = "success", records: int = 0, detail: str = "",
             target_user: str = "", incident: dict | None = None) -> None:
        self.events.append(
            dict(ts=ts, user_id=user.user_id, event_type=event_type, ip=ip, country=country,
                 asn_type=self.asn_of(ip), device_id=device, session_id=session_id, status=status,
                 records=int(records), detail=detail, target_user=target_user,
                 incident_id=incident["incident_id"] if incident else "",
                 scenario=incident["scenario"] if incident else "")
        )

    # ---------- normal behaviour ----------
    def _login(self, ts, u, ip, country, dev, incident=None, mfa=True) -> str:
        sid = self._session()
        if mfa:
            self.emit(ts - timedelta(seconds=20), u, "mfa_challenge", ip, country, dev, status="approved", incident=incident)
        self.emit(ts, u, "login", ip, country, dev, session_id=sid, incident=incident)
        u.sessions.append((sid, ts, ip))
        return sid

    def _work_block(self, ts, u, sid, ip, country, dev, hours, export_scale=1.0, incident=None, access_scale=1.0):
        n_access = self.rng.poisson(8 * hours / 4 * access_scale) + 1
        for _ in range(n_access):
            t = ts + timedelta(minutes=float(self.rng.uniform(1, hours * 60)))
            self.emit(t, u, "record_access", ip, country, dev, sid,
                      records=int(self.rng.integers(1, 40) * access_scale), incident=incident)
        n_exp = self.rng.poisson(EXPORT_LAMBDA[u.role] * hours / 8)
        for _ in range(n_exp):
            t = ts + timedelta(minutes=float(self.rng.uniform(1, hours * 60)))
            self.emit(t, u, "export", ip, country, dev, sid,
                      records=int(np.exp(self.rng.normal(u.export_mu, 0.5)) * export_scale), incident=incident)
        if u.role == "engineer":
            for _ in range(self.rng.poisson(2.0 * hours / 8)):
                t = ts + timedelta(minutes=float(self.rng.uniform(1, hours * 60)))
                self.emit(t, u, "api_query", ip, country, dev, sid,
                          records=int(self.rng.integers(50, 1500)), incident=incident)
        if u.is_admin:
            for _ in range(self.rng.poisson(0.4)):
                t = ts + timedelta(minutes=float(self.rng.uniform(1, hours * 60)))
                tgt = self.users[int(self.rng.integers(self.n_users))].user_id
                self.emit(t, u, "permission_grant", ip, country, dev, sid,
                          detail=str(self.rng.choice(NORMAL_PERMS)), target_user=tgt, incident=incident)
            if self.rng.random() < 0.03:
                self.emit(ts + timedelta(minutes=30), u, "auth_policy_change", ip, country, dev, sid,
                          detail=str(self.rng.choice(SAFE_POLICIES)), incident=incident)
        self.emit(ts + timedelta(hours=hours), u, "logout", ip, country, dev, sid, incident=incident)

    def normal_day(self, day: int, u: User, override: dict | None = None) -> None:
        """Simulate a regular working day. `override` lets benign scenarios tweak context."""
        o = override or {}
        weekend = (START + timedelta(days=day)).weekday() >= 5
        if not o and self.rng.random() > (0.07 if weekend else 0.93):
            return
        country = o.get("country", u.country)
        dev = o.get("device", u.devices[0] if self.rng.random() < 0.85 else u.devices[-1])
        n_sessions = 1 + int(self.rng.random() < 0.4)
        hour = u.start_hour + self.rng.normal(0, 0.7)
        for s in range(n_sessions):
            ip = o.get("ip") or (u.corp_ip_for(self) if self.rng.random() < 0.55 else u.home_ip)
            ts = self._utc(day, hour, u.country)
            if not o.get("skip_typo") and self.rng.random() < 0.04:  # fat-fingered password
                self.emit(ts - timedelta(minutes=1), u, "login", ip, country, dev, status="failure")
            sid = self._login(ts, u, ip, country, dev, incident=o.get("incident"))
            hours = float(self.rng.uniform(2.5, 4.5))
            self._work_block(ts, u, sid, ip, country, dev, hours,
                             export_scale=o.get("export_scale", 1.0), incident=o.get("incident"))
            hour += hours + float(self.rng.uniform(0.5, 1.5))

    # ---------- incidents ----------
    def _new_incident(self, scenario: str, u: User, day: int, stealth: float) -> dict:
        inc = dict(incident_id=f"INC-{next(self._inc_counter):04d}", scenario=scenario, user_id=u.user_id,
                   day=day, malicious=scenario in ATTACKS, stealth=round(stealth, 3))
        self.incidents.append(inc)
        return inc

    def _attacker_location(self, u: User, stealth: float) -> tuple[str, str]:
        if self.rng.random() < 0.25 + 0.5 * stealth:  # stealthy attackers proxy through the victim's country
            country = u.country
        else:
            country = str(self.rng.choice(FOREIGN_ATTACK_COUNTRIES))
        asn = "residential" if self.rng.random() < 0.3 + 0.5 * stealth else "hosting"
        return self._ip(asn), country

    def inject_attack(self, scenario: str, day: int, u: User) -> None:
        r = self.rng
        st = float(r.beta(2, 2))  # stealth in [0, 1]
        inc = self._new_incident(scenario, u, day, st)
        hour = float(r.uniform(0, 22)) if r.random() < 0.6 else u.start_hour + r.uniform(0, 8)
        ts = self._utc(day, hour, u.country)
        self.normal_day(day, u)  # the victim usually still works that day

        if scenario == "brute_force":
            ip, cc = self._attacker_location(u, st)
            dev = self._device()
            n_fail = int(4 + (1 - st) * 40 + r.integers(0, 6))
            for k in range(n_fail):
                self.emit(ts + timedelta(seconds=float(k * r.uniform(2, 40))), u, "login", ip, cc, dev,
                          status="failure", incident=inc)
            if r.random() < 0.6:
                t2 = ts + timedelta(minutes=n_fail + 2)
                sid = self._login(t2, u, ip, cc, dev, incident=inc, mfa=r.random() < 0.5)
                for _ in range(r.integers(2, 10)):
                    self.emit(t2 + timedelta(minutes=float(r.uniform(1, 30))), u, "record_access", ip, cc, dev, sid,
                              records=int(r.integers(10, 300)), incident=inc)

        elif scenario == "account_takeover":
            ip, cc = self._attacker_location(u, st)
            dev = self._device()
            sid = self._login(ts, u, ip, cc, dev, incident=inc)
            mult = 1 + (1 - st) * 12
            for _ in range(int(r.integers(5, 25))):
                self.emit(ts + timedelta(minutes=float(r.uniform(1, 90))), u, "record_access", ip, cc, dev, sid,
                          records=int(r.integers(10, 60) * mult), incident=inc)
            if r.random() < 0.6:
                self.emit(ts + timedelta(minutes=95), u, "export", ip, cc, dev, sid,
                          records=int(np.exp(u.export_mu) * (1 + (1 - st) * 8)), incident=inc)

        elif scenario == "privilege_escalation":
            dev = u.devices[0]
            ip = u.corp_ip_for(self) if r.random() < 0.5 else u.home_ip
            sid = self._login(ts, u, ip, u.country, dev, incident=inc)
            perm = str(r.choice(HIGH_RISK_PERMS))
            target = u.user_id if r.random() < 0.75 else self.users[int(r.integers(self.n_users))].user_id
            self.emit(ts + timedelta(minutes=3), u, "permission_grant", ip, u.country, dev, sid,
                      detail=perm, target_user=target, incident=inc)
            for _ in range(int(r.integers(3, 15))):
                self.emit(ts + timedelta(minutes=float(r.uniform(5, 120))), u, "record_access", ip, u.country, dev,
                          sid, records=int(r.integers(50, 400) * (1 + (1 - st) * 4)), incident=inc)

        elif scenario == "insider_exfiltration":
            dev = u.devices[0]
            ip = u.home_ip if r.random() < 0.6 else u.corp_ip_for(self)
            sid = self._login(ts, u, ip, u.country, dev, incident=inc)
            mult = 4 + (1 - st) * 40
            for _ in range(int(r.integers(2, 10))):
                self.emit(ts + timedelta(minutes=float(r.uniform(2, 120))), u, "export", ip, u.country, dev, sid,
                          records=int(np.exp(r.normal(u.export_mu, 0.4)) * mult), incident=inc)

        elif scenario == "token_abuse":
            dev = u.devices[0]
            sid = self._login(ts, u, u.home_ip, u.country, dev, incident=inc)
            self.emit(ts + timedelta(minutes=2), u, "token_create", u.home_ip, u.country, dev, sid,
                      detail="personal_access_token", incident=inc)
            ip, cc = self._attacker_location(u, st)
            ip = self._ip("hosting") if r.random() < 0.8 else ip
            n_q = int(10 + (1 - st) * 150)
            for _ in range(n_q):
                self.emit(ts + timedelta(minutes=float(r.uniform(10, 240))), u, "api_query", ip, cc, "api-client",
                          "", records=int(r.integers(200, 2000)), detail="token_auth", incident=inc)

        elif scenario == "mfa_fatigue":
            ip, cc = self._attacker_location(u, st)
            dev = self._device()
            n_push = int(3 + (1 - st) * 15)
            for k in range(n_push):
                self.emit(ts + timedelta(minutes=k * float(r.uniform(0.5, 3))), u, "mfa_challenge", ip, cc, dev,
                          status="denied", incident=inc)
            t2 = ts + timedelta(minutes=n_push * 2 + 1)
            sid = self._login(t2, u, ip, cc, dev, incident=inc)
            for _ in range(int(r.integers(3, 15))):
                self.emit(t2 + timedelta(minutes=float(r.uniform(1, 60))), u, "record_access", ip, cc, dev, sid,
                          records=int(r.integers(10, 200)), incident=inc)

        elif scenario == "session_hijack":
            # Victim logs in normally, attacker replays the session cookie from elsewhere.
            dev = u.devices[0]
            t0 = self._utc(day, u.start_hour + 0.5, u.country)
            sid = self._login(t0, u, u.home_ip, u.country, dev, incident=inc)
            ip, cc = self._attacker_location(u, st * 0.5)
            t1 = t0 + timedelta(minutes=float(r.uniform(10, 90)))
            for _ in range(int(r.integers(4, 20))):
                self.emit(t1 + timedelta(minutes=float(r.uniform(0, 60))), u, "record_access", ip, cc, dev, sid,
                          records=int(r.integers(10, 300)), incident=inc)
            if r.random() < 0.5:
                self.emit(t1 + timedelta(minutes=61), u, "export", ip, cc, dev, sid,
                          records=int(np.exp(u.export_mu) * (2 + (1 - st) * 6)), incident=inc)

        elif scenario == "auth_policy_tamper":
            ip, cc = (u.corp_ip_for(self), u.country) if r.random() < st else self._attacker_location(u, st)
            dev = u.devices[0] if r.random() < st else self._device()
            sid = self._login(ts, u, ip, cc, dev, incident=inc)
            self.emit(ts + timedelta(minutes=4), u, "auth_policy_change", ip, cc, dev, sid,
                      detail=str(r.choice(RISKY_POLICIES)), incident=inc)
            if r.random() < 0.5:
                self.emit(ts + timedelta(minutes=8), u, "permission_grant", ip, cc, dev, sid,
                          detail=str(r.choice(HIGH_RISK_PERMS)), target_user=u.user_id, incident=inc)

    def inject_benign(self, scenario: str, day: int, u: User) -> None:
        r = self.rng
        inc = self._new_incident(scenario, u, day, 0.0)
        if scenario == "legit_travel":
            cc = str(r.choice(TRAVEL_COUNTRIES))
            self.normal_day(day, u, dict(country=cc, ip=self._ip("residential" if r.random() < 0.6 else "mobile"),
                                         device=u.devices[0], incident=inc))
        elif scenario == "quarter_end_export":
            self.normal_day(day, u, dict(export_scale=float(r.uniform(2, 6)), incident=inc, ip=u.corp_ip_for(self)))
            ts = self._utc(day, u.start_hour + 3, u.country)
            for _ in range(int(r.integers(2, 6))):
                self.emit(ts + timedelta(minutes=float(r.uniform(0, 200))), u, "export", u.corp_ip_for(self),
                          u.country, u.devices[0], records=int(np.exp(u.export_mu) * r.uniform(2, 8)), incident=inc)
        elif scenario == "forgot_password":
            ts = self._utc(day, u.start_hour, u.country)
            for k in range(int(r.integers(3, 9))):
                self.emit(ts + timedelta(seconds=30 * k), u, "login", u.home_ip, u.country, u.devices[0],
                          status="failure", incident=inc)
            self.normal_day(day, u, dict(ip=u.home_ip, device=u.devices[0], incident=inc, skip_typo=True))
        elif scenario == "new_laptop":
            dev = self._device()
            u.devices.append(dev)
            self.normal_day(day, u, dict(device=dev, ip=self._ip("residential"), incident=inc))
        elif scenario == "admin_onboarding":
            ts = self._utc(day, u.start_hour + 1, u.country)
            ip = u.corp_ip_for(self)
            sid = self._login(ts, u, ip, u.country, u.devices[0], incident=inc)
            for _ in range(int(r.integers(3, 10))):
                perm = str(r.choice(HIGH_RISK_PERMS + NORMAL_PERMS * 2))
                tgt = self.users[int(r.integers(self.n_users))].user_id
                self.emit(ts + timedelta(minutes=float(r.uniform(1, 90))), u, "permission_grant", ip, u.country,
                          u.devices[0], sid, detail=perm, target_user=tgt, incident=inc)
        elif scenario == "vpn_user":
            self.normal_day(day, u, dict(ip=self._ip("hosting"), device=u.devices[0], incident=inc,
                                         country=u.country if r.random() < 0.6 else str(r.choice(TRAVEL_COUNTRIES))))
        elif scenario == "engineer_new_token":
            ts = self._utc(day, u.start_hour + 1, u.country)
            ip = u.corp_ip_for(self)
            sid = self._login(ts, u, ip, u.country, u.devices[0], incident=inc)
            self.emit(ts + timedelta(minutes=5), u, "token_create", ip, u.country, u.devices[0], sid,
                      detail="personal_access_token", incident=inc)
            ci_ip = self._ip("hosting")  # CI runner in the cloud
            for _ in range(int(r.integers(10, 60))):
                self.emit(ts + timedelta(minutes=float(r.uniform(10, 300))), u, "api_query", ci_ip, u.country,
                          "api-client", records=int(r.integers(100, 1500)), detail="token_auth", incident=inc)

    # ---------- driver ----------
    def run(self, n_days: int, warmup: int, attacks_per_day: float, benign_per_day: float) -> None:
        r = self.rng
        attack_names, benign_names = list(ATTACKS), list(BENIGN)
        for day in range(n_days):
            weekend = (START + timedelta(days=day)).weekday() >= 5
            touched: set[str] = set()
            plan: list[tuple[str, str, User]] = []
            if day >= warmup and not weekend:
                for _ in range(r.poisson(attacks_per_day)):
                    plan.append(("attack", str(r.choice(attack_names)), None))
                for _ in range(r.poisson(benign_per_day)):
                    plan.append(("benign", str(r.choice(benign_names)), None))
            for kind, scen, _ in plan:
                u = self._eligible_user(scen, touched)
                touched.add(u.user_id)
                (self.inject_attack if kind == "attack" else self.inject_benign)(scen, day, u)
            for u in self.users:
                if u.user_id not in touched:
                    self.normal_day(day, u)

    def _eligible_user(self, scenario: str, touched: set[str]) -> User:
        need_role = {"auth_policy_tamper": "admin", "admin_onboarding": "admin", "engineer_new_token": "engineer"}
        pool = [u for u in self.users if u.user_id not in touched]
        if scenario in need_role:
            pool = [u for u in pool if u.role == need_role[scenario]]
        if scenario == "privilege_escalation":
            pool = [u for u in pool if not u.is_admin]
        if scenario == "quarter_end_export":
            pool = [u for u in pool if u.role in ("sales", "analyst")]
        return pool[int(self.rng.integers(len(pool)))]


def _corp_ip_for(self: User, sim: LogSimulator) -> str:
    return sim.corp_ips[self.country]


User.corp_ip_for = _corp_ip_for  # type: ignore[attr-defined]


def generate(n_users: int = 400, seed: int = 7, attacks_per_day: float = 6.0, benign_per_day: float = 6.0):
    sim = LogSimulator(n_users, seed)
    sim.run(config.N_DAYS, config.WARMUP_DAYS, attacks_per_day, benign_per_day)
    events = pd.DataFrame(sim.events).sort_values("ts", kind="stable").reset_index(drop=True)
    events.insert(0, "event_id", [f"e{i:07d}" for i in range(len(events))])
    users = pd.DataFrame([dict(user_id=u.user_id, role=u.role, country=u.country, is_admin=u.is_admin,
                               tz_offset=TZ_OFFSET[u.country]) for u in sim.users])
    incidents = pd.DataFrame(sim.incidents)
    return events, users, incidents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    events, users, incidents = generate(args.users, args.seed)
    events.to_parquet(config.EVENTS_PATH, index=False)
    users.to_parquet(config.USERS_PATH, index=False)
    incidents.to_parquet(config.INCIDENTS_PATH, index=False)
    print(f"events={len(events):,} users={len(users)} incidents={len(incidents)} "
          f"(malicious={int(incidents.malicious.sum())}, benign look-alikes={int((~incidents.malicious).sum())})")
    print(events.event_type.value_counts().to_string())


if __name__ == "__main__":
    main()
