#!/usr/bin/env bash
# Tail stocks-agent logs
set -euo pipefail

SERVICE_NAME="algotrading-stocks"
LINES="${1:-50}"

echo "=== stocks-agent logs (last ${LINES} lines) ==="
echo "Press Ctrl+C to stop following"
echo ""

journalctl -u ${SERVICE_NAME} --no-pager -n "${LINES}" -f
