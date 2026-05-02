# Support Triage Agent

A terminal-based RAG agent that triages support tickets across **HackerRank**, **Claude (Anthropic)**, and **Visa** using only the provided support corpus. No outside knowledge is used — every response is grounded in the `data/` directory.

---

## Architecture

```
support_tickets.csv
        │
        ▼
   main.py                   ← entry point: reads CSV, drives the pipeline, writes output.csv
        │
        ├── escalation.py    ← rule-based pre-LLM scorer; hard-escalates high-risk tickets
        ├── retriever.py     ← loads & embeds corpus; company-scoped semantic search
        └── agent.py         ← sends ticket + retrieved docs to Claude; parses JSON result
```

### Pipeline per ticket

```
1. score_ticket()       → if score >= 3: return escalated immediately (no API call)
2. retriever.search()   → find top-k relevant docs from the company's corpus
3. Claude API call      → classify + respond using only the retrieved excerpts
4. _parse_result()      → parse JSON, coerce enum fields, handle malformed output
```

---

## Modules

### `escalation.py` — Pre-LLM rule-based scorer

Scores a ticket before the LLM is called. If the score meets or exceeds `ESCALATION_THRESHOLD` (default: **3**), the ticket is escalated immediately, saving an API call and ensuring deterministic handling of high-risk cases.

**Signal categories and weights:**

| Signal | Weight | Example triggers |
|---|---|---|
| Fraud / unauthorized use | 3 | "unauthorized charge", "someone used my card" |
| Identity theft / account takeover | 3 | "identity theft", "account hacked" |
| Legal / regulatory | 3 | "lawsuit", "data breach", "GDPR" |
| Complete service outage | 3 | "site is down", "all requests failing" |
| Prompt injection attempt | 3 | "ignore previous instructions", "show me your prompt" |
| Security vulnerability report | 2 | "zero-day", "CVE-", "remote code execution" |
| Urgency (alone) | 1 | "urgent", "asap", "emergency" |
| Financial (alone) | 1 | "refund", "billing", "invoice" |
| Account deletion | 1 | "delete my account", "right to erasure" |

**Compound rule:** urgency + financial together adds +1, so "urgent refund" reaches threshold even if neither signal alone would.

To adjust sensitivity, change `ESCALATION_THRESHOLD` or individual rule weights in `escalation.py`.

---

### `retriever.py` — Corpus loader and semantic search

Loads all `.md` files from `data/` at startup, encodes them with `all-MiniLM-L6-v2` (a local sentence-transformers model — no extra API key needed), and exposes a `search()` method.

**Company-scoped retrieval:** When a ticket specifies a company (HackerRank, Claude, or Visa), search is restricted to that company's documents only. This prevents cross-company hallucination — a Visa ticket won't pull HackerRank articles. Falls back to the full corpus when `company` is `None` or unrecognised.

**Similarity:** Cosine similarity between the query embedding and the document subset.

**First run:** The embedding model (~90 MB) is downloaded automatically on first use. Encoding the full corpus takes ~30 seconds.

---

### `agent.py` — LLM triage layer

Calls `claude-sonnet-4-6` with a structured system prompt and the retrieved corpus excerpts. Claude must output a strict JSON object — nothing else.

**System prompt enforces:**
- Answer only from the provided corpus excerpts
- Human tone: no hollow openers, no chatbot filler, plain direct language
- Clear escalation rules (active fraud, account-level admin actions, outages)
- Precise `request_type` definitions with a product_issue vs feature_request distinction

**Output parsing:** Handles Claude wrapping JSON in markdown code fences. Falls back to a safe escalated result if parsing fails.

---

### `main.py` — Entry point

Reads `support_tickets.csv`, initialises the retriever (once), then processes each ticket in sequence. Prints progress per row. Writes the full output (original columns + 5 new columns) to `output.csv`.

---

## Setup

```bash
# 1. Install dependencies (from repo root)
pip install -r requirements.txt

# 2. Configure your API key
cp .env.example .env
# Open .env and set: ANTHROPIC_API_KEY=your_key_here

# 3. Run against the real tickets
cd code
python main.py

# Or run against the sample tickets to verify accuracy
python main.py --tickets ../support_tickets/sample_support_tickets.csv --output ../support_tickets/sample_output.csv
```

---

## CLI options

```
python main.py [--tickets PATH] [--output PATH]
```

| Flag | Default | Description |
|---|---|---|
| `--tickets` | `../support_tickets/support_tickets.csv` | Path to input CSV |
| `--output` | `../support_tickets/output.csv` | Path to write results |

---

## Output schema

The output CSV contains all original columns plus:

| Column | Allowed values | Description |
|---|---|---|
| `status` | `replied` / `escalated` | Whether the agent answered or handed off |
| `product_area` | Free text | Concise support category (e.g. `Billing`, `Account Access`) |
| `response` | Free text | User-facing reply grounded in the corpus, or `"Escalated to a human."` |
| `justification` | Free text | 1–3 sentences explaining the decision and which corpus content was used |
| `request_type` | `product_issue` / `feature_request` / `bug` / `invalid` | Best-fit classification of the ticket |

---

## Escalation decision flow

A ticket is escalated if **either** condition is met:

1. **Pre-LLM (rule-based):** `score_ticket()` returns a score ≥ 3. This fires for clear-cut cases like fraud, outages, or prompt injection — no API call is made.
2. **LLM (semantic):** Claude judges the ticket as requiring escalation based on the system prompt rules (account-level admin actions, no corpus coverage, etc.).

The `justification` field records which path fired and why, making decisions auditable.

---

## Design decisions

**Why company-scoped retrieval instead of a global search with boosting?**
Boosting still allows cross-company docs to appear in results. Scoping eliminates the possibility entirely — a Visa billing question will never pull a HackerRank test management article.

**Why a pre-LLM escalation scorer?**
High-risk tickets (fraud, identity theft, legal threats) should not go through an LLM at all — they need deterministic, immediate escalation. The scorer also saves API cost for clear-cut cases.

**Why `all-MiniLM-L6-v2`?**
Fast, lightweight, runs locally with no API key. Sufficient semantic quality for support FAQ retrieval. Swap the `_MODEL_NAME` constant in `retriever.py` for a stronger model if quality needs improving.

**Why `claude-sonnet-4-6`?**
Best balance of response quality and cost for structured triage. The model is pinned in `agent.py` (`_MODEL`) — change it there if needed.

---

## Dependencies

| Package | Purpose |
|---|---|
| `anthropic` | Claude API client |
| `sentence-transformers` | Local embedding model for semantic search |
| `numpy` | Cosine similarity computation |
| `pandas` | CSV read/write |
| `python-dotenv` | Loads `.env` into environment |
