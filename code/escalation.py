"""
Rule-based escalation scorer.

Assigns a score to a ticket based on keyword and pattern signals.
If the score meets or exceeds ESCALATION_THRESHOLD, the ticket is
escalated immediately without calling the LLM.
"""

import re

ESCALATION_THRESHOLD = 3

# Each entry: (compiled pattern, score, reason label)
_RULES: list[tuple[re.Pattern, int, str]] = [
    # Fraud / financial crime
    (re.compile(r"\b(fraud|fraudulent|scam|stolen card|unauthorized (charge|transaction|payment))\b", re.I), 3, "fraud"),
    # Identity / account takeover
    (re.compile(r"\b(identity theft|account (hacked|compromised|taken over)|someone else.{0,20}account)\b", re.I), 3, "identity_theft"),
    # Legal / regulatory
    (re.compile(r"\b(lawyer|attorney|lawsuit|legal action|sue|subpoena|gdpr|data breach|compliance violation)\b", re.I), 3, "legal"),
    # Security vulnerabilities (reporting, not asking)
    (re.compile(r"\b(security (vulnerability|exploit|breach)|zero.?day|CVE-\d|remote code execution|RCE)\b", re.I), 2, "security_vuln"),
    # Complete outages
    (re.compile(r"\b(site (is )?down|nothing (is )?working|all (pages?|requests?|services?) (are )?(inaccessible|failing|broken))\b", re.I), 3, "outage"),
    # Prompt injection attempts
    (re.compile(r"\b(ignore (previous|all|prior) instructions?|disregard (your )?(rules?|instructions?)|show (me )?(your )?(prompt|system|internal))\b", re.I), 3, "prompt_injection"),
    # Urgency + financial together (weaker signal alone, stronger combined)
    (re.compile(r"\b(urgent|emergency|immediately|asap|right now)\b", re.I), 1, "urgency"),
    (re.compile(r"\b(refund|chargeback|money|payment|billing|invoice)\b", re.I), 1, "financial"),
    # Sensitive account actions
    (re.compile(r"\b(delete (my )?(account|data)|permanently (remove|delete)|right to (erasure|be forgotten))\b", re.I), 1, "account_deletion"),
]


class EscalationScore:
    def __init__(self, score: int, reasons: list[str], escalate: bool):
        self.score = score
        self.reasons = reasons
        self.escalate = escalate

    def __repr__(self) -> str:
        return f"EscalationScore(score={self.score}, escalate={self.escalate}, reasons={self.reasons})"


def score_ticket(issue: str, subject: str = "") -> EscalationScore:
    """Score a ticket and decide whether to hard-escalate before the LLM."""
    text = f"{subject} {issue}"
    total = 0
    reasons: list[str] = []
    seen: set[str] = set()

    for pattern, weight, label in _RULES:
        if pattern.search(text) and label not in seen:
            total += weight
            reasons.append(label)
            seen.add(label)

    # Compound rule: urgency + financial together tips over threshold
    if "urgency" in seen and "financial" in seen and total < ESCALATION_THRESHOLD:
        total += 1  # push compound signal over if borderline

    return EscalationScore(
        score=total,
        reasons=reasons,
        escalate=total >= ESCALATION_THRESHOLD,
    )
