#!/usr/bin/env bash
# Install the Pi camera gateway as a systemd service.
set -euo pipefail

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this as the normal Pi user, not with sudo. Example: ./install_camera_service.sh \$USER"
  exit 1
fi

PI_USER="${1:?Usage: ./install_camera_service.sh <pi-username>}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="/opt/threat-edge"
CAMERA_HOST="${CAMERA_HOST:-0.0.0.0}"
CAMERA_PORT="${CAMERA_PORT:-8000}"
CAMERA_WIDTH="${CAMERA_WIDTH:-1280}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-720}"
CAMERA_FPS="${CAMERA_FPS:-15}"
CAMERA_JPEG_QUALITY="${CAMERA_JPEG_QUALITY:-80}"

sudo apt update
sudo apt install -y --no-install-recommends python3-picamera2 python3-opencv python3-flask python3-waitress
sudo install -d -m 0755 "$TARGET_DIR"
sudo install -m 0755 "$SOURCE_DIR/pi_camera_mjpeg.py" "$TARGET_DIR/pi_camera_mjpeg.py"
sudo install -d -m 0755 /etc/threat-edge
sudo tee /etc/threat-edge/camera.env >/dev/null <<'EOF'
CAMERA_HOST=0.0.0.0
CAMERA_PORT=8000
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
CAMERA_FPS=15
CAMERA_JPEG_QUALITY=80
EOF

sudo tee /etc/systemd/system/threat-edge-camera.service >/dev/null <<EOF
[Unit]
Description=Threat Detection Pi Camera Gateway
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${PI_USER}
Group=${PI_USER}
SupplementaryGroups=video
EnvironmentFile=/etc/threat-edge/camera.env
ExecStart=/usr/bin/python3 /opt/threat-edge/pi_camera_mjpeg.py --host ${CAMERA_HOST} --port ${CAMERA_PORT} --width ${CAMERA_WIDTH} --height ${CAMERA_HEIGHT} --fps ${CAMERA_FPS} --jpeg-quality ${CAMERA_JPEG_QUALITY}
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

sudo usermod -aG video "$PI_USER"
sudo systemctl daemon-reload
sudo systemctl enable --now threat-edge-camera.service
echo "Installed. Reboot or sign out/in once so '${PI_USER}' receives the video group."
echo "Check status: sudo systemctl status threat-edge-camera.service"
