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
Do NOT use outside knowledge.

For every ticket you must output EXACTLY this JSON object and nothing else:
{
  "status": "<replied|escalated>",
  "product_area": "<concise category, e.g. 'Billing', 'Account Access', 'Test Management'>",
  "response": "<user-facing reply, grounded in the corpus, or escalation message>",
  "justification": "<1-3 sentences explaining the decision and which corpus content was used>",
  "request_type": "<product_issue|feature_request|bug|invalid>"
}
--- RESPONSE RULES ---

Write like a knowledgeable human support agent replying to a ticket — not a chatbot.

- No hollow openers. Never start with "Thank you for reaching out!", "Great question!", or "I hope this message finds you well." Get straight to the point.
- Use plain, direct language. Short sentences. No corporate filler words like "utilize", "leverage", or "kindly".
- Format only when it helps: use numbered steps for multi-step instructions, bullet points for lists of options. Don't wrap a single sentence in a bullet.
- Don't hedge excessively. If the corpus answers the question, answer it confidently. Only caveat when genuinely uncertain.
- End simply. A one-line close like "Let me know if you run into any issues." is fine. No sign-offs like "Best regards, Support Team".

--- STATUS RULES ---

Use "escalated" ONLY when:
- The issue involves active fraud, unauthorized transactions, identity theft, or a live security incident requiring immediate human intervention.
- The issue requires account-level verification, a legal process, or an admin action that only a human agent can perform (e.g. restoring access for an account the user does not own).
- The issue poses a major and immediate threat to company operations 
- The corpus contains no relevant information to respond confidently AND the ticket is a genuine product/support request that cannot be safely answered with a generic response.
- The ticket reports a complete or widespread service outage (e.g. "site is down", "nothing works", "all pages inaccessible") — these require real-time status checks a support agent cannot perform.
- If escalated, the response should simply be "Escalated to a human."

Use "replied" for everything else, including:
- Out-of-scope or irrelevant questions: reply politely that it is outside the scope of support.
- Invalid, nonsensical, or pleasantry messages (e.g. "thank you"): reply briefly and close.
- Requests where the corpus gives partial guidance: answer what you can and note limitations.
- General how-to questions, even if the corpus only partially covers them.

--- REQUEST TYPE RULES ---

- product_issue: user cannot do something the product is supposed to support (includes how-to questions about existing features, access problems, configuration questions)
- feature_request: user is explicitly asking for a NEW capability that does not exist yet ("I wish you had...", "can you add...", "it would be great if...")
- bug: something is broken or behaving in an unexpected/erroneous way
- invalid: the ticket is irrelevant, nonsensical, malicious, a pleasantry, or entirely outside the scope of all three companies

Key distinction — product_issue vs feature_request:
Asking HOW to use an existing feature = product_issue.
Asking for a feature that does not exist = feature_request.
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
