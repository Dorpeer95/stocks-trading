#!/usr/bin/env bash
# Restart stocks-agent safely
set -euo pipefail

SERVICE_NAME="algotrading-stocks"

echo "=== Restarting stocks-agent ==="

# Stop
echo "Stopping ${SERVICE_NAME}..."
sudo systemctl stop ${SERVICE_NAME}
sleep 2

# Verify stopped
if sudo systemctl is-active --quiet ${SERVICE_NAME}; then
    echo "⚠️  Service still running, force stopping..."
    sudo systemctl kill ${SERVICE_NAME}
    sleep 2
fi

# Start
echo "Starting ${SERVICE_NAME}..."
sudo systemctl start ${SERVICE_NAME}
sleep 5

# Verify running
if sudo systemctl is-active --quiet ${SERVICE_NAME}; then
    echo "✅ ${SERVICE_NAME} is running"
else
    echo "❌ ${SERVICE_NAME} failed to start"
    sudo journalctl -u ${SERVICE_NAME} --no-pager -n 20
    exit 1
fi

# Health check
sleep 5
if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Health check passed"
    curl -sf http://localhost:8001/health | python3 -m json.tool
else
    echo "⚠️  Health check failed (may need more time)"
fi
