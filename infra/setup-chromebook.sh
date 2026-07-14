#!/usr/bin/env bash
# setup-chromebook.sh
#
# Run once on the Chromebook Linux terminal to install Docker,
# start the monitoring stack, and set up the Fritz collector.
#
# Prerequisites:
#   - Linux (Crostini) enabled in ChromeOS settings
#   - Repo cloned:  git clone <repo-url> ~/pylon && cd ~/pylon/infra
#   - .env.local present at repo root with FRITZ_PASSWORD set

set -e

INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$INFRA_DIR")"
ENV_FILE="$REPO_ROOT/.env.local"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║          Pylon — Chromebook Monitoring Stack Setup       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Load env ───────────────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found. Create it with FRITZ_PASSWORD=yourpassword"
  exit 1
fi
set -a; source "$ENV_FILE"; set +a
echo "✓ Loaded $ENV_FILE"

# ── 2. Install Docker if missing ──────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo ""
  echo "▶  Installing Docker..."
  sudo apt-get update -qq
  sudo apt-get install -y ca-certificates curl gnupg lsb-release

  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg

  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/debian $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

  sudo apt-get update -qq
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
  sudo usermod -aG docker "$USER"
  echo "✓ Docker installed"
  echo ""
  echo "NOTE: Docker group membership takes effect in new shells."
  echo "      If the next step fails, run:  newgrp docker"
  newgrp docker || true
else
  echo "✓ Docker already installed"
fi

# ── 3. Install Python deps for collector ──────────────────────────────────────
echo ""
echo "▶  Installing collector dependencies..."
sudo apt-get install -y python3-pip python3-venv -qq
python3 -m venv "$INFRA_DIR/collector/.venv"
"$INFRA_DIR/collector/.venv/bin/pip" install -q \
  -r "$INFRA_DIR/collector/requirements.txt"
echo "✓ Collector dependencies installed"

# ── 4. Start Docker stack ─────────────────────────────────────────────────────
echo ""
echo "▶  Starting Prometheus + Grafana + Pi-hole..."
cd "$INFRA_DIR"
docker compose up -d
echo "✓ Stack started"

# ── 5. Start collector as background service ──────────────────────────────────
echo ""
echo "▶  Starting Fritz collector on :9101..."

# Write a small systemd user unit so it survives reboots
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/fritz-collector.service <<EOF
[Unit]
Description=Fritz TR-064 Prometheus Collector
After=network.target

[Service]
EnvironmentFile=$ENV_FILE
ExecStart=$INFRA_DIR/collector/.venv/bin/python $INFRA_DIR/collector/fritz_collector.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable fritz-collector
systemctl --user start fritz-collector
echo "✓ Collector running (systemd user service)"

# ── 6. Summary ────────────────────────────────────────────────────────────────
echo ""
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "<tailscale-ip>")
echo "══════════════════════════════════════════════════════════════"
echo "  Stack is up. Access from any device on Tailscale:"
echo ""
echo "  Grafana    http://$TAILSCALE_IP:3000   (admin / admin)"
echo "  Prometheus http://$TAILSCALE_IP:9090"
echo "  Pi-hole    http://$TAILSCALE_IP:8080"
echo "  Collector  http://$TAILSCALE_IP:9101/metrics"
echo "══════════════════════════════════════════════════════════════"
echo ""
