"""
Wipe all data from every table in the Supabase `stocks` schema.

Usage:
    python scripts/clean_db.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from agent.persistence import init_supabase, _table

TABLES = [
    "portfolio_log",
    "signal_history",
    "portfolio_holdings",
    "equity_snapshots",
    "trades",
    "positions",
    "opportunities",
    "market_events",
    "gpt_briefings",
    "model_versions",
    "daily_scores",
    "universe",
]


# Tables with bigint id columns need numeric comparison
BIGINT_ID_TABLES = {"equity_snapshots", "trades", "positions", "opportunities", "market_events", "model_versions", "daily_scores"}
# Tables with no id column — use a different column for the filter
SPECIAL_PK = {
    "universe": "ticker",
}


def clean_all():
    init_supabase()
    print("Cleaning all tables in stocks schema...\n")

    for table in TABLES:
        try:
            if table in SPECIAL_PK:
                col = SPECIAL_PK[table]
                result = _table(table).delete().neq(col, "").execute()
            elif table in BIGINT_ID_TABLES:
                result = _table(table).delete().gt("id", 0).execute()
            else:
                result = _table(table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            count = len(result.data) if result.data else 0
            print(f"  {table}: deleted {count} rows")
        except Exception as e:
            print(f"  {table}: FAILED — {e}")

    print("\nDone. All tables wiped.")


if __name__ == "__main__":
    clean_all()
