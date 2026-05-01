# Support Triage Agent

Terminal-based RAG agent that triages support tickets across HackerRank, Claude, and Visa using only the provided corpus.

## Architecture

```
support_tickets.csv
      │
      ▼
main.py          ← entry point; reads CSV, writes output.csv
      │
      ├── retriever.py   ← loads data/ corpus, embeds with sentence-transformers, top-k search
      └── agent.py       ← sends ticket + retrieved docs to Claude, parses structured JSON result
```

**Retrieval:** `all-MiniLM-L6-v2` embeddings over all `.md` files in `data/`. Same-company docs get a 1.2× score boost.

**Generation:** `claude-sonnet-4-6` with a strict system prompt constraining answers to the retrieved corpus. Outputs JSON with five fields.

**Escalation logic:** Fraud, security incidents, missing corpus coverage, malicious/invalid requests → `escalated`.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# 3. Run
cd code
python main.py
```

Output is written to `support_tickets/output.csv`.

## Output columns

| Column | Values |
|--------|--------|
| `status` | `replied` / `escalated` |
| `product_area` | Free-text category (e.g. `Billing`, `Account Access`) |
| `response` | User-facing reply grounded in the corpus |
| `justification` | Short explanation of the decision |
| `request_type` | `product_issue` / `feature_request` / `bug` / `invalid` |

## Options

```
python main.py --tickets <path> --output <path>
```

## Dependencies

- `anthropic` — Claude API
- `sentence-transformers` — local embedding model (no extra API key needed)
- `pandas` — CSV I/O
- `python-dotenv` — `.env` loading
