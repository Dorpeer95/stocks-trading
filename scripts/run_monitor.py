import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv
load_dotenv()

from agent.agent import AgentLoop

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s")

def main():
    print("🚀 Starting Intraday Monitor...")
    agent = AgentLoop()
    agent.intraday_monitor()
    print("✅ Intraday Monitor finished.")

if __name__ == "__main__":
    main()
