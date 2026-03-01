#!/usr/bin/env bash
# =============================================================
# Setup stocks-agent on a DigitalOcean droplet
# Run as root or with sudo on a fresh droplet
#
# Prerequisites:
#   - Ubuntu 22.04+
#   - Python 3.11+ installed
#   - Git installed
#   - algotrading user exists (from crypto bot setup)
# =============================================================
set -euo pipefail

PROJECT_DIR="/home/algotrading/stocks-agent"
SERVICE_NAME="algotrading-stocks"
REPO_URL="https://github.com/Dorpeer95/stocks-trading.git"
PYTHON_BIN="python3.11"

echo "=== Setting up stocks-agent ==="

# ---------------------------------------------------------------------------
# 1. Clone repository
# ---------------------------------------------------------------------------
if [ -d "$PROJECT_DIR" ]; then
    echo "Project directory exists, pulling latest..."
    cd "$PROJECT_DIR"
    sudo -u algotrading git pull origin main
else
    echo "Cloning repository..."
    sudo -u algotrading git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Creating Python virtual environment..."
    sudo -u algotrading $PYTHON_BIN -m venv "$PROJECT_DIR/venv"
fi

echo "Installing dependencies..."
sudo -u algotrading "$PROJECT_DIR/venv/bin/pip" install --upgrade pip
sudo -u algotrading "$PROJECT_DIR/venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 3. Create logs directory
# ---------------------------------------------------------------------------
sudo -u algotrading mkdir -p "$PROJECT_DIR/logs"

# ---------------------------------------------------------------------------
# 4. Create .env file (if not exists)
# ---------------------------------------------------------------------------
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "Creating .env from .env.example..."
    sudo -u algotrading cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "⚠️  IMPORTANT: Edit $PROJECT_DIR/.env with your actual keys!"
fi

# ---------------------------------------------------------------------------
# 5. Create systemd service
# ---------------------------------------------------------------------------
echo "Creating systemd service..."
cat > /etc/systemd/system/${SERVICE_NAME}.service << 'EOF'
[Unit]
Description=Stocks Trading Advisory Bot
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=algotrading
Group=algotrading
WorkingDirectory=/home/algotrading/stocks-agent
Environment=PATH=/home/algotrading/stocks-agent/venv/bin:/usr/bin
EnvironmentFile=/home/algotrading/stocks-agent/.env
ExecStart=/home/algotrading/stocks-agent/venv/bin/python main.py
Restart=on-failure
RestartSec=30
StartLimitBurst=5
StartLimitIntervalSec=300

# Memory limits
MemoryMax=450M
MemoryHigh=400M

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=stocks-agent

[Install]
WantedBy=multi-user.target
EOF

# ---------------------------------------------------------------------------
# 6. Configure log rotation
# ---------------------------------------------------------------------------
cat > /etc/logrotate.d/stocks-agent << EOF
/home/algotrading/stocks-agent/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 algotrading algotrading
}
EOF

# ---------------------------------------------------------------------------
# 7. Enable and start service
# ---------------------------------------------------------------------------
echo "Enabling and starting service..."
systemctl daemon-reload
systemctl enable ${SERVICE_NAME}
systemctl start ${SERVICE_NAME}

# Wait for startup
sleep 5

# Check status
if systemctl is-active --quiet ${SERVICE_NAME}; then
    echo "✅ ${SERVICE_NAME} is running"
else
    echo "❌ ${SERVICE_NAME} failed to start"
    journalctl -u ${SERVICE_NAME} --no-pager -n 20
    exit 1
fi

# Check health
sleep 10
if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "⚠️  Health endpoint not ready yet (may need more time)"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Useful commands:"
echo "  systemctl status ${SERVICE_NAME}"
echo "  journalctl -u ${SERVICE_NAME} -f"
echo "  curl http://localhost:8001/health"
echo "  curl http://localhost:8001/status"
echo ""
echo "⚠️  Don't forget to edit: $PROJECT_DIR/.env"
