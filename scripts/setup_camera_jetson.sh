#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR=/opt/robot-car

if [[ ${EUID} -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 2
fi

if ! id robot >/dev/null 2>&1; then
  echo "Service account 'robot' is missing; run scripts/setup_jetson.sh first." >&2
  exit 4
fi

camera_groups=video
if getent group plugdev >/dev/null 2>&1; then
  camera_groups=video,plugdev
fi
usermod -a -G "${camera_groups}" robot

missing=()
for module in cv2 numpy pyrealsense2; do
  if ! /usr/bin/python3 -c "import ${module}" >/dev/null 2>&1; then
    missing+=("${module}")
  fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Missing system-Python modules: ${missing[*]}" >&2
  echo "Install the JetPack-compatible librealsense/pyrealsense2 and OpenCV packages first." >&2
  echo "The service deliberately does not install an incompatible desktop RealSense wheel." >&2
  exit 3
fi

install -d -o robot -g robot "${INSTALL_DIR}"
cp -a "${SOURCE_DIR}/jetson_camera" "${INSTALL_DIR}/"
chown -R robot:robot "${INSTALL_DIR}/jetson_camera"
cat > /etc/default/robot-car-camera <<EOF
CAMERA_LISTEN=0.0.0.0
CAMERA_PORT=8080
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=15
CAMERA_OUTPUT_WIDTH=320
CAMERA_OUTPUT_HEIGHT=240
CAMERA_PUBLISH_FPS=3
CAMERA_JPEG_QUALITY=65
EOF
install -m 0644 "${SOURCE_DIR}/jetson_camera/robot-car-camera.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable robot-car-camera.service
systemctl restart robot-car-camera.service
echo "Camera server installed at http://JETSON_IP:8080/status"
