"""
Support triage agent.

Given one ticket (issue, subject, company) it:
1. Retrieves the top-k most relevant corpus documents.
2. Asks Claude to classify and respond using only those documents.
3. Returns a structured TicketResult.
"""

import os
from dataclasses import dataclass

import anthropic

from escalation import score_ticket
from retriever import Retriever

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
_MODEL = "claude-sonnet-4-6"
_TOP_K = 6

SYSTEM_PROMPT = """You are a support triage agent for HackerRank, Claude (Anthropic), and Visa.
Answer ONLY using the corpus excerpts in the user message. No outside knowledge.

Output EXACTLY this JSON, nothing else:
{"status":"replied|escalated","product_area":"concise category","response":"user-facing reply or 'Escalated to a human.'","justification":"1-3 sentences on decision and corpus used","request_type":"product_issue|feature_request|bug|invalid"}

--- RESPONSE STYLE ---
Write as a human support agent, not a chatbot.
- No hollow openers. Never start with phrases like "Thank you for reaching out", "Great question", "Absolutely!", "Of course!", "Happy to help", or "I hope this helps". Get straight to the point.
- Plain, direct language. No filler words (utilize, leverage, kindly).
- Numbered steps for instructions; bullets only for true lists. Not for single sentences.
- Answer confidently when corpus is clear. Caveat only when genuinely uncertain.
- One-line close is fine. No sign-offs.

--- STATUS ---
Use "escalated" only when:
- Active fraud, unauthorized transactions, identity theft, or live security incident.
- Requires account verification, legal process, or admin action only a human can do.
- Major, immediate threat to company operations.
- Complete/widespread service outage (needs real-time status checks).
- No corpus coverage AND cannot be safely answered with a generic reply.
Response for escalated tickets must be exactly: "Escalated to a human."

Use "replied" for everything else: out-of-scope questions, pleasantries, partial corpus coverage, how-to questions.

--- REQUEST TYPE ---
- product_issue: can't do something the product supports; how-to questions; access/config problems
- feature_request: explicitly asking for a new capability that doesn't exist yet
- bug: broken behaviour, unexpected errors, or security/vulnerability reports
- invalid: irrelevant, nonsensical, malicious, or pleasantry; out of scope for all three companies
Note: asking HOW to use an existing feature = product_issue, not feature_request.
"""


@dataclass
class TicketResult:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str


def process_ticket(
    issue: str,
    subject: str,
    company: str,
    retriever: Retriever,
) -> TicketResult:
    esc = score_ticket(issue=issue, subject=subject)
    if esc.escalate:
        return TicketResult(
            status="escalated",
            product_area="Escalated",
            response="Escalated to a human.",
            justification=f"Pre-LLM escalation score {esc.score} >= threshold. Signals: {', '.join(esc.reasons)}.",
            request_type="product_issue",
        )

    docs = retriever.search(query=f"{subject} {issue}", company=company or None, top_k=_TOP_K)
    corpus_block = _format_corpus(docs)

    user_message = f"""Company: {company or 'Unknown'}
Subject: {subject or '(none)'}
Issue: {issue}

--- SUPPORT CORPUS EXCERPTS ---
{corpus_block}
--- END CORPUS ---

Now output the JSON triage result."""

    message = _client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    raw = message.content[0].text.strip()
    return _parse_result(raw)


def _format_corpus(docs: list[tuple]) -> str:
    parts = []
    for i, (doc, score) in enumerate(docs, 1):
        # Truncate very long docs to avoid blowing the context window
        excerpt = doc.text[:1500]
        parts.append(f"[{i}] (source: {doc.path}, score: {score:.2f})\n{excerpt}")
    return "\n\n".join(parts)


def _parse_result(raw: str) -> TicketResult:
    import json
    # Strip markdown code fences if Claude wrapped the JSON
    text = raw
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(text)
        return TicketResult(
            status=_coerce(data.get("status", "escalated"), {"replied", "escalated"}, "escalated"),
            product_area=str(data.get("product_area", "General")),
            response=str(data.get("response", "")),
            justification=str(data.get("justification", "")),
            request_type=_coerce(
                data.get("request_type", "product_issue"),
                {"product_issue", "feature_request", "bug", "invalid"},
                "product_issue",
            ),
        )
    except (json.JSONDecodeError, KeyError):
        return TicketResult(
            status="escalated",
            product_area="General",
            response="Unable to parse agent response. Escalating for manual review.",
            justification=f"Raw output could not be parsed: {raw[:200]}",
            request_type="product_issue",
        )


def _coerce(value: str, allowed: set[str], default: str) -> str:
    return value if value in allowed else default
