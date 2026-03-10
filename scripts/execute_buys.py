import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv()

from agent.portfolio import execute_buy_opportunities
from agent.persistence import get_pending_opportunities

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s")

def main():
    print("🚀 Fetching pending opportunities...")
    opps = get_pending_opportunities()
    if not opps:
        print("ℹ️ No pending opportunities found.")
        return
    
    print(f"✅ Found {len(opps)} opportunities. Executing buys...")
    execute_buy_opportunities(opps)
    print("✅ Execution finished.")

if __name__ == "__main__":
    main()
