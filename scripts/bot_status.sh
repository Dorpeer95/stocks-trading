#!/usr/bin/env bash
# Check stocks-agent status on DigitalOcean
set -euo pipefail

SERVICE_NAME="algotrading-stocks"

echo "=== stocks-agent status ==="
echo ""

# Service status
echo "--- Service ---"
systemctl status ${SERVICE_NAME} --no-pager 2>/dev/null || echo "Service not found"
echo ""

# Health endpoint
echo "--- Health ---"
curl -sf http://localhost:8001/health 2>/dev/null | python3 -m json.tool || echo "Health endpoint unreachable"
echo ""

# Status endpoint
echo "--- Bot Status ---"
curl -sf http://localhost:8001/status 2>/dev/null | python3 -m json.tool || echo "Status endpoint unreachable"
echo ""

# Memory usage
echo "--- Memory ---"
ps aux | head -1
ps aux | grep "stocks-agent" | grep -v grep || echo "No stocks-agent process found"
echo ""
echo "Total system memory:"
free -h
echo ""

# Crypto bot status (coexistence check)
echo "--- Crypto bot coexistence ---"
systemctl is-active algotrading-crypto 2>/dev/null && echo "Crypto bot: RUNNING" || echo "Crypto bot: NOT RUNNING"
