"""Weights of the additive trust algorithm of Bradatsch et al. (TrustCom 2023, arXiv 2402.08299, Eqs. 1 to 3).

Source: github.com/zintpavowj/ZT_Score-based_Network_Level_AC_PDP_TA at commit 46e099bf95518ff4f887f0abdedb0f6e8baf073a,
internal/app/algorithm/additive/additive_trust.go (line numbers below are that file's unless another file is named).
The spec fixes them ("the weights are those of the reference implementation", D80), so they are part of the algorithm
and not settings. How the reference's attributes map to "agent i sends to j" is in CONTRACT.md.

Where this feature departs from the reference, and why (each is also an open question in CONTRACT.md):
  * Decision. The paper grants iff TS > RL (Eq. 3) and the spec keeps it (D80); the reference code grants on
    TS >= RL (additive.go 157). C reports RL - TS and denies when it exceeds the calibrated threshold (core contract),
    so a tie at the threshold is granted, as in the reference code.
  * Enterprise presence. The reference adds 3.0 when the last access is at most 24 * 7 hours ago (405) and 2.0 more
    when the user is "expected" (410, weight 28). Only the first part is in the spec's table (rounds since i was last
    active); its window is the setting conventional.c.presence_window_rounds.
  * Network threat level. The reference weighs the level by a table for levels 1 to 10 and gives 0 for any other value
    (747 to 756). C saturates at level 10: a larger number of flagged agents must not lower the risk.
  * The reference counts the network threat weight twice in its dynamic risk score, once as "system threat level"
    (277, 726 to 734) and once as "network threat level" (283, 747 to 756), because both read
    risk_system_network_threat_levels[NetworkThreatLevel]. The spec lists one attribute, so C counts it once.
"""
import torch

# ---- trust attributes (sum of the weights whose attribute value is in its target set: Eq. 1)
AUTHENTICATION = 8.0       # 34  device_cert_auth: the delivery carries a valid mTLS credential or message signature
                           #     (user_faceid_auth, 19, weighs the same, so the choice between them changes nothing)
CHANNEL = 5.0 + 5.0 + 5.0  # 68, 71, 74  channel authentication SHA256-RSA, confidentiality AES, integrity SHA256-RSA
CONSTANT_TRUST = AUTHENTICATION + CHANNEL      # attributes that are constant in these experiments (spec 2.3)
SERVICE_USAGE = 3.0        # 24  user_service_usage; rule 339-346: the service is in the usage history
ACCESS_RATE = 5.0          # 22  user_access_rate; rule 383-397: min <= current <= max of the history, empty: 0
PRESENCE = 3.0             # 27  user_enterprise_presence_time; rule 404-407: time since last access <= window
INPUT_BEHAVIOUR = 3.0      # 21  user_input_behavior; rule 323-336: min <= current <= max of the history, empty: 0
TRUST_HISTORY = 5.0        # 31  user_trust_history; rule 428-441: the partial score is within [min, max] of the
                           #     stored scores (floatlib.MinMax, floatlib.go 8-25: empty history gives 0), no window,
                           #     no decay. calculateUserTrustScore (145-178) calls it last, with the sum of the others.

# ---- risk attributes (Eq. 2)
# 130-139  risk_system_network_threat_levels[1..10]; level 0 has no entry, so it weighs 0 (lookup 747-756). The table
# is indexed by the level, clamped to its last entry (see the departure above).
NETWORK_THREAT = torch.tensor([0.0, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 12.0, 15.0, 20.0, 25.0], dtype=torch.float64)
MAX_THREAT_LEVEL = len(NETWORK_THREAT) - 1
