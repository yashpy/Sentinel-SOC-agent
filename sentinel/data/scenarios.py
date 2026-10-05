"""Attack and benign look-alike scenario catalog (the ground truth for evaluation)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Scenario:
    name: str
    malicious: bool
    description: str
    techniques: tuple[str, ...] = ()  # accepted MITRE ATT&CK technique IDs (parent or sub)
    acceptable_actions: tuple[str, ...] = ("none", "monitor")
    best_action: str = "none"
    tags: tuple[str, ...] = field(default_factory=tuple)


ATTACKS: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario(
            "brute_force",
            True,
            "Many failed logins from an unfamiliar IP, sometimes followed by a successful login.",
            ("T1110",),
            ("reset_password", "enforce_mfa", "revoke_sessions", "freeze_user"),
            "reset_password",
        ),
        Scenario(
            "account_takeover",
            True,
            "Successful login from a new country/device/IP followed by unusual data access.",
            ("T1078",),
            ("revoke_sessions", "reset_password", "freeze_user"),
            "revoke_sessions",
        ),
        Scenario(
            "privilege_escalation",
            True,
            "Non-admin user grants a high-risk permission to themselves, then uses it.",
            ("T1098",),
            ("remove_permission", "freeze_user"),
            "remove_permission",
        ),
        Scenario(
            "insider_exfiltration",
            True,
            "User exports far more records than their baseline from their usual device and network.",
            ("T1213", "T1567", "T1530", "T1020", "T1119"),
            ("freeze_user", "revoke_sessions"),
            "freeze_user",
        ),
        Scenario(
            "token_abuse",
            True,
            "New API token created, then bulk API queries from a hosting-provider IP.",
            ("T1528", "T1550"),
            ("revoke_sessions", "freeze_user", "reset_password"),
            "revoke_sessions",
        ),
        Scenario(
            "mfa_fatigue",
            True,
            "Burst of denied MFA push challenges, then an approval and login from a new location.",
            ("T1621",),
            ("reset_password", "revoke_sessions", "freeze_user", "enforce_mfa"),
            "reset_password",
        ),
        Scenario(
            "session_hijack",
            True,
            "Existing session reused from a distant IP without a login (impossible travel).",
            ("T1539", "T1550", "T1185"),
            ("revoke_sessions", "reset_password", "freeze_user"),
            "revoke_sessions",
        ),
        Scenario(
            "auth_policy_tamper",
            True,
            "Admin account weakens authentication controls (disables MFA, adds trusted IP range).",
            ("T1556", "T1685"),  # T1685 replaced the revoked T1562 in current ATT&CK releases
            ("freeze_user", "revoke_sessions", "reset_password"),
            "freeze_user",
        ),
    ]
}

BENIGN: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario("legit_travel", False, "User travels and logs in from another country on a known device."),
        Scenario("quarter_end_export", False, "Large but legitimate report exports at quarter end."),
        Scenario("forgot_password", False, "Several failed logins from a known IP, then success."),
        Scenario("new_laptop", False, "New device and new IP in the user's home country."),
        Scenario("admin_onboarding", False, "Admin grants permission sets to several other users."),
        Scenario("vpn_user", False, "User connects through a commercial VPN (hosting-provider IP)."),
        Scenario("engineer_new_token", False, "Engineer creates an API token and runs bulk queries."),
    ]
}

ALL_SCENARIOS: dict[str, Scenario] = {**ATTACKS, **BENIGN}
VERDICT_SCENARIOS = list(ATTACKS) + ["benign"]


def technique_match(predicted: str | None, scenario: str) -> bool:
    """True if the predicted ATT&CK ID matches (parent-level) one of the scenario's techniques."""
    if not predicted or scenario not in ATTACKS:
        return False
    parent = predicted.strip().upper().split(".")[0]
    return parent in ATTACKS[scenario].techniques
