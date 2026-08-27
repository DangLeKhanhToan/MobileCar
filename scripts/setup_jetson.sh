#!/usr/bin/env bash
set -euo pipefail

# Install the TCP-to-Arduino bridge as a dedicated, least-privilege service.
# Run from this repository: sudo ./scripts/setup_jetson.sh [serial-device]

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR=/opt/robot-car
SERVICE_USER=robot
SERIAL_DEVICE="${1:-}"

if [[ ${EUID} -ne 0 ]]; then
  echo "Run as root: sudo $0 [serial-device]" >&2
  exit 2
fi

# Publish the Jetson's hostname as <hostname>.local for Windows/macOS/Linux mDNS clients.
if ! command -v avahi-daemon >/dev/null 2>&1; then
  apt-get update
  apt-get install -y avahi-daemon
fi
systemctl enable --now avahi-daemon.service

if [[ -z ${SERIAL_DEVICE} ]]; then
  mapfile -t candidates < <(find /dev -maxdepth 1 \( -name 'ttyACM*' -o -name 'ttyUSB*' \) -print 2>/dev/null | sort)
  if [[ ${#candidates[@]} -ne 1 ]]; then
    echo "Expected exactly one Arduino serial device; found ${#candidates[@]}." >&2
    echo "Plug in the Uno, then pass it explicitly, for example: sudo $0 /dev/ttyACM0" >&2
    exit 3
  fi
  SERIAL_DEVICE="${candidates[0]}"
fi

if [[ ! -c ${SERIAL_DEVICE} ]]; then
  echo "Not a character device: ${SERIAL_DEVICE}" >&2
  exit 4
fi

if ! id "${SERVICE_USER}" >/dev/null 2>&1; then
  useradd --system --create-home --groups dialout "${SERVICE_USER}"
else
  usermod -a -G dialout "${SERVICE_USER}"
fi

install -d -o "${SERVICE_USER}" -g "${SERVICE_USER}" "${INSTALL_DIR}"
cp -a "${SOURCE_DIR}/jetson_bridge" "${INSTALL_DIR}/"
chown -R "${SERVICE_USER}:${SERVICE_USER}" "${INSTALL_DIR}"

if [[ ! -x ${INSTALL_DIR}/.venv/bin/python ]]; then
  sudo -u "${SERVICE_USER}" python3 -m venv "${INSTALL_DIR}/.venv"
fi
sudo -u "${SERVICE_USER}" "${INSTALL_DIR}/.venv/bin/python" -m pip install \
  --disable-pip-version-check -r "${INSTALL_DIR}/jetson_bridge/requirements.txt"

# Prefer a stable udev name when the USB adapter exposes identifying data.
properties="$(udevadm info --query=property --name="${SERIAL_DEVICE}" 2>/dev/null || true)"
vendor="$(sed -n 's/^ID_VENDOR_ID=//p' <<<"${properties}" | head -n1)"
model="$(sed -n 's/^ID_MODEL_ID=//p' <<<"${properties}" | head -n1)"
serial="$(sed -n 's/^ID_SERIAL_SHORT=//p' <<<"${properties}" | head -n1)"
service_serial="${SERIAL_DEVICE}"
if [[ -n ${vendor} && -n ${model} && -n ${serial} ]]; then
  printf 'SUBSYSTEM=="tty", ATTRS{idVendor}=="%s", ATTRS{idProduct}=="%s", ATTRS{serial}=="%s", SYMLINK+="robot_base", GROUP="dialout", MODE="0660"\n' \
    "${vendor}" "${model}" "${serial}" > /etc/udev/rules.d/99-robot-base.rules
  udevadm control --reload-rules
  udevadm trigger
  service_serial=/dev/robot_base
else
  echo "Warning: device has no stable USB serial identity; using ${SERIAL_DEVICE}." >&2
fi

cat > /etc/default/robot-car-bridge <<EOF
ROBOT_SERIAL=${service_serial}
ROBOT_BAUD=115200
ROBOT_LISTEN=0.0.0.0
ROBOT_PORT=8765
EOF
install -m 0644 "${SOURCE_DIR}/jetson_bridge/robot-car-bridge.service" /etc/systemd/system/robot-car-bridge.service
systemctl daemon-reload
systemctl enable robot-car-bridge.service
systemctl restart robot-car-bridge.service

if command -v ufw >/dev/null 2>&1 && ufw status | grep -q '^Status: active'; then
  ufw allow 8765/tcp comment 'RobotCar bridge'
fi

sleep 1
if ! systemctl is-active --quiet robot-car-bridge.service; then
  echo "robot-car-bridge failed to start. Recent service log:" >&2
  journalctl -u robot-car-bridge.service -n 30 --no-pager >&2
  exit 5
fi
if command -v ss >/dev/null 2>&1 && ! ss -ltn | grep -q ':8765 '; then
  echo "robot-car-bridge is active but TCP port 8765 is not listening." >&2
  journalctl -u robot-car-bridge.service -n 30 --no-pager >&2
  exit 6
fi

echo "Installed robot-car-bridge on TCP port 8765 using ${service_serial}."
echo "mDNS address: $(hostname).local (Windows may require Bonjour for .local resolution)."
echo "Next: systemctl status robot-car-bridge && python3 scripts/robot_link_test.py JETSON_IP"
