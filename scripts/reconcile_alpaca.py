import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv()

from agent.persistence import get_open_positions
from utils.alpaca_broker import alpaca_client, execute_trade

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("reconcile")

def reconcile(close_zombies=False):
    if not alpaca_client:
        logger.error("Alpaca client not initialized. Check your credentials in .env")
        return

    # 1. Fetch Alpaca positions
    try:
        alpaca_positions = alpaca_client.list_positions()
        alpaca_map = {p.symbol: p for p in alpaca_positions}
    except Exception as e:
        logger.error(f"Failed to fetch Alpaca positions: {e}")
        return

    # 2. Fetch Supabase positions
    db_positions = get_open_positions()
    db_map = {p["ticker"]: p for p in db_positions}

    logger.info("=" * 40)
    logger.info(f"Checking {len(alpaca_map)} Alpaca positions vs {len(db_map)} DB entries")
    logger.info("=" * 40)

    # 3. Identify discrepancies
    zombies = []
    for ticker, info in alpaca_map.items():
        if ticker not in db_map:
            zombies.append(ticker)

    missing = []
    for ticker in db_map:
        if ticker not in alpaca_map:
            missing.append(ticker)

    # 4. Report
    if not zombies and not missing:
        logger.info("✅ Portfolios are perfectly aligned.")
    else:
        if zombies:
            logger.info(f"💀 Found {len(zombies)} ZOMBIE positions (on Alpaca, not in DB):")
            for z in zombies:
                qty = alpaca_map[z].qty
                val = alpaca_map[z].market_value
                logger.info(f"  - {z}: {qty} shares (Value: ${val})")
        
        if missing:
            logger.info(f"❓ Found {len(missing)} MISSING positions (in DB, not on Alpaca):")
            for m in missing:
                logger.info(f"  - {m}")

    # 5. Action
    if close_zombies and zombies:
        logger.info("-" * 40)
        logger.info(f"Liquidating {len(zombies)} zombie positions on Alpaca...")
        for z in zombies:
            qty = float(alpaca_map[z].qty)
            if qty > 0:
                logger.info(f"  Selling {qty} of {z}...")
                execute_trade(z, "sell", qty, "market")
            elif qty < 0:
                logger.info(f"  Buying {abs(qty)} to close short on {z}...")
                execute_trade(z, "buy", abs(qty), "market")
        logger.info("Done.")
    elif zombies:
        logger.info("-" * 40)
        logger.info("Run with --close-zombies to automatically liquidate these.")

def main():
    parser = argparse.ArgumentParser(description="Reconcile Alpaca positions with Supabase")
    parser.add_argument("--close-zombies", action="store_true", help="Close positions on Alpaca not found in DB")
    args = parser.parse_args()
    
    reconcile(close_zombies=args.close_zombies)

if __name__ == "__main__":
    main()
