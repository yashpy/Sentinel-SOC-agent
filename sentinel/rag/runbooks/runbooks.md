# RB-001 | Password guessing and credential stuffing | T1110
Signals: a burst of failed logins for one account from a single unfamiliar IP, often a hosting provider or foreign country; sometimes a successful login right after the failures. Low-and-slow variants spread a handful of failures over time.
Benign look-alike: a user who forgot their password fails 3-8 times from their *known* home IP and device, then succeeds.
Response: if the guessing succeeded, reset the password and revoke active sessions; if it only failed, enforce MFA and monitor. Block the source IP at the edge.

# RB-002 | Compromised valid account / account takeover | T1078
Signals: a *successful* login from a never-seen IP, device and often country, with MFA approved (phished or token-replayed credentials), followed by record access volume well above the user's baseline or an export.
Benign look-alikes: legitimate travel on a known device; a new laptop in the home country with normal activity volume.
Response: revoke all sessions first (fast, reversible), then force a password reset. Freeze the user only if data was exported.

# RB-003 | Privilege escalation via permission self-grant | T1098
Signals: a non-admin user grants a high-risk permission (Modify_All_Data, View_All_Data, Manage_Users, Export_Reports, API_Enabled) to themselves or a colluding account and then accesses many records.
Benign look-alike: an administrator onboarding new hires grants permission sets to *other* users during business hours.
Response: remove the granted permission immediately, review the audit trail of records touched, then interview the user.

# RB-004 | Insider data exfiltration via bulk report export | T1213 T1567 T1530
Signals: the user exports far more records than their own historical baseline (5x-50x), from their usual device and network, often outside working hours or close to a resignation date.
Benign look-alike: quarter-end reporting by sales or analysts, 2x-6x baseline from the corporate network during business hours.
Response: freeze the user to stop further exports, preserve the export logs as evidence, and notify legal/HR.

# RB-005 | Stolen or abused API access token | T1528 T1550
Signals: a new personal access token is created, then hundreds of API queries authenticated by that token arrive from a hosting-provider IP that the user has never used, in a different country.
Benign look-alike: an engineer creates a token for a CI pipeline that runs from a cloud IP in the home country at moderate volume.
Response: revoke the token and all sessions, rotate credentials, and check for data pulled through the API.

# RB-006 | MFA fatigue / push bombing | T1621
Signals: many *denied* MFA push challenges in a short window from an unfamiliar IP, followed by one approval and a login from that IP.
Benign look-alike: a single accidental denial followed by approval from the user's own device.
Response: reset the password (the attacker already knows it), revoke sessions, switch the user to number-matching MFA.

# RB-007 | Session cookie theft / session hijacking | T1539 T1550 T1185
Signals: activity on an existing session ID from an IP and country different from the IP that created the session, with no new login event; physically impossible travel between events minutes apart.
Benign look-alike: a mobile user switching between Wi-Fi and cellular inside the same country.
Response: revoke the session(s) immediately, reset the password, and check the endpoint for infostealer malware.

# RB-008 | Weakening authentication controls | T1556 T1685
Signals: an admin account disables MFA, adds a trusted IP range, or relaxes the password policy, often from a new IP or device and sometimes followed by a self-grant.
Benign look-alike: changing session timeout or login hours through the normal change process.
Response: freeze the admin account, revert the policy change, and review every change made in that session.

# RB-009 | Triage policy and response tiers
Read-only actions (none, monitor) need no approval. Reversible actions (revoke_sessions, reset_password, enforce_mfa) execute after the verifier passes. Destructive actions (remove_permission, freeze_user) always require human approval. Never take a containment action when the evidence supports a benign explanation; choose monitor instead and document why.
