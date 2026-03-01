#!/usr/bin/env bash
# Deploy stocks-agent to DigitalOcean (run from local machine)
set -euo pipefail

# Configuration — update these
DO_HOST="${DO_HOST:-your-droplet-ip}"
DO_USER="${DO_USER:-algotrading}"
DEPLOY_DIR="/home/algotrading/stocks-agent"
SERVICE_NAME="algotrading-stocks"

echo "=== Deploying stocks-agent to ${DO_HOST} ==="

ssh "${DO_USER}@${DO_HOST}" << ENDSSH
    set -e
    cd ${DEPLOY_DIR}
    git pull origin main
    source venv/bin/activate
    pip install -r requirements.txt --quiet
    sudo systemctl restart ${SERVICE_NAME}
    sleep 5
    if sudo systemctl is-active --quiet ${SERVICE_NAME}; then
        echo "✅ Deployed and running"
    else
        echo "❌ Deploy failed"
        sudo journalctl -u ${SERVICE_NAME} --no-pager -n 20
        exit 1
    fi
ENDSSH

echo "=== Deploy complete ==="
