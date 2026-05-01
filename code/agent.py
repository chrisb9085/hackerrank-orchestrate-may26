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

from retriever import Retriever

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
_MODEL = "claude-sonnet-4-6"
_TOP_K = 6

SYSTEM_PROMPT = """You are a support triage agent for three companies: HackerRank, Claude (Anthropic), and Visa.
You must answer ONLY using the support corpus excerpts provided in the user message.
Do NOT use outside knowledge. If the corpus does not contain enough information to answer safely, escalate.

For every ticket you must output EXACTLY this JSON object and nothing else:
{
  "status": "<replied|escalated>",
  "product_area": "<concise category, e.g. 'Billing', 'Account Access', 'Test Management'>",
  "response": "<user-facing reply, grounded in the corpus, or escalation message>",
  "justification": "<1-3 sentences explaining the decision and which corpus content was used>",
  "request_type": "<product_issue|feature_request|bug|invalid>"
}

Escalate when:
- The issue involves fraud, unauthorized transactions, or security incidents.
- The issue requires account verification or admin-level action you cannot perform.
- The corpus does not contain relevant information to answer reliably.
- The request is malicious, abusive, or clearly out of scope.

Classify request_type as:
- product_issue: user cannot do something the product should support
- feature_request: user wants something new or different
- bug: something is broken or behaving unexpectedly
- invalid: irrelevant, malicious, or nonsensical request
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
        system=SYSTEM_PROMPT,
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
