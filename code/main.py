"""
Entry point — reads support_tickets/support_tickets.csv, runs each ticket
through the triage agent, and writes support_tickets/output.csv.

Usage:
    python main.py
    python main.py --tickets ../support_tickets/support_tickets.csv --output ../support_tickets/output.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load .env before importing agent (which reads ANTHROPIC_API_KEY at import time)
load_dotenv(Path(__file__).parent.parent / ".env")

from agent import TicketResult, process_ticket
from retriever import Retriever

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_TICKETS = _REPO_ROOT / "support_tickets" / "support_tickets.csv"
_DEFAULT_OUTPUT = _REPO_ROOT / "support_tickets" / "output.csv"

OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]


def main() -> None:
    parser = argparse.ArgumentParser(description="HackerRank Orchestrate — support triage agent")
    parser.add_argument("--tickets", default=str(_DEFAULT_TICKETS), help="Path to input CSV")
    parser.add_argument("--output", default=str(_DEFAULT_OUTPUT), help="Path to output CSV")
    args = parser.parse_args()

    tickets_path = Path(args.tickets)
    output_path = Path(args.output)

    if not tickets_path.exists():
        print(f"Error: tickets file not found: {tickets_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading corpus and building index (first run may take ~30 s)...")
    retriever = Retriever()
    print(f"Corpus loaded: {len(retriever._docs)} documents.\n")

    df = pd.read_csv(tickets_path)
    _validate_columns(df, tickets_path)

    results: list[dict] = []
    total = len(df)
    for i, row in df.iterrows():
        issue = str(row.get("Issue") or row.get("issue") or "")
        subject = str(row.get("Subject") or row.get("subject") or "")
        company = str(row.get("Company") or row.get("company") or "")

        print(f"[{i+1}/{total}] {company or 'Unknown'}: {subject[:60] or issue[:60]}")

        try:
            result: TicketResult = process_ticket(
                issue=issue,
                subject=subject,
                company=company,
                retriever=retriever,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            result = TicketResult(
                status="escalated",
                product_area="General",
                response="An error occurred while processing this ticket. Escalating for manual review.",
                justification=f"Processing error: {exc}",
                request_type="product_issue",
            )

        results.append({
            "status": result.status,
            "product_area": result.product_area,
            "response": result.response,
            "justification": result.justification,
            "request_type": result.request_type,
        })
        print(f"  -> {result.status} | {result.request_type} | {result.product_area}")

    out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    out_df.to_csv(output_path, index=False)
    print(f"\nDone. Output written to {output_path}")


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    cols_lower = [c.lower() for c in df.columns]
    if "issue" not in cols_lower:
        print(f"Warning: no 'Issue' column found in {path}. Columns: {list(df.columns)}", file=sys.stderr)


if __name__ == "__main__":
    main()
