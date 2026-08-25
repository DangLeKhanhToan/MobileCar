"""Simple RobotCar manual and RGB-D person-avoidance test application."""

import sys
import time
import socket
import json
import math
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import serial
from serial.tools import list_ports
from PyQt6.QtCore import QEvent, QSettings, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QBoxLayout, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SPEED_STATES = {"0": 0, "4": 100, "6": 155, "7": 180, "8": 200, "9": 230, "q": 255}
MOTION_LABELS = {
    "S": "STOP", "F": "Forward", "B": "Reverse", "L": "Pivot left", "R": "Pivot right",
    "G": "Forward-left", "I": "Forward-right", "H": "Reverse-left", "J": "Reverse-right",
}


@dataclass
class PersonObservation:
    detected: bool = False
    distance_m: float | None = None
    center_x: float = 0.0
    velocity_x_px_s: float = 0.0
    confidence: float = 0.0
    depth_available: bool = False
    obstacle_distance_m: float | None = None
    policy_motion: str = "S"
    policy_speed: str = "0"
    policy_confidence: float = 0.0
    policy_reason: str = "SocialWalker unavailable"
    people_xz: list | None = None
    obstacles_xz: list | None = None
    trajectories: list | None = None
    trajectory_scores: list | None = None
    selected_trajectory: int = -1


class ModelPlot(QWidget):
    """Top-down view of model inputs, candidate paths, and ranked output."""

    goal_clicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.observation = PersonObservation()
        self.robot_pose = (0.0, 0.0, 0.0)
        self.goal = None
        self.trail = [(0.0, 0.0)]
        self.setMinimumHeight(180)

    def set_observation(self, observation):
        self.observation = observation
        self.update()

    def set_navigation(self, pose, goal, trail):
        self.robot_pose, self.goal, self.trail = pose, goal, list(trail)
        self.update()

    def _map_geometry(self):
        margin, max_z, max_x = 24.0, 8.5, 7.0
        sx = (self.width() - 2 * margin) / (2 * max_x)
        sy = (self.height() - 2 * margin) / max_z
        return margin, max_z, max_x, sx, sy

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            margin, max_z, max_x, sx, sy = self._map_geometry()
            x = (event.position().x() - self.width() / 2) / sx
            y = (self.height() - margin - event.position().y()) / sy
            if -max_x <= x <= max_x and 0.0 <= y <= max_z:
                self.goal_clicked.emit(float(x), float(y))

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#101318"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        margin, max_z, max_x, sx, sy = self._map_geometry()

        def point(x, z):
            return int(self.width() / 2 + x * sx), int(self.height() - margin - z * sy)

        painter.setPen(QPen(QColor("#39414d"), 1))
        for z in range(0, 9, 2):
            x1, y = point(-max_x, z); x2, _ = point(max_x, z)
            painter.drawLine(x1, y, x2, y); painter.drawText(2, y + 4, f"{z}m")
        px, py, heading = self.robot_pose

        def world(local_x, local_z):
            return (px + math.cos(heading) * local_x - math.sin(heading) * local_z,
                    py + math.sin(heading) * local_x + math.cos(heading) * local_z)

        painter.setPen(QPen(QColor("#8793a5"), 2))
        for a, b in zip(self.trail, self.trail[1:]):
            painter.drawLine(*point(*a), *point(*b))
        painter.setPen(QPen(QColor("#bec7d5"), 2)); painter.setBrush(QColor("#101318"))
        rx, ry = point(px, py); painter.drawEllipse(rx - 6, ry - 6, 12, 12)
        hx, hy = point(*world(0.0, 0.45)); painter.drawLine(rx, ry, hx, hy)
        if self.goal is not None:
            gx, gy = point(*self.goal)
            painter.setPen(QPen(QColor("#ff4fd8"), 3)); painter.drawEllipse(gx - 8, gy - 8, 16, 16)
            painter.drawLine(gx - 11, gy, gx + 11, gy); painter.drawLine(gx, gy - 11, gx, gy + 11)
        paths = self.observation.trajectories or []
        scores = self.observation.trajectory_scores or []
        selected = self.observation.selected_trajectory
        for index, path in enumerate(paths):
            chosen = index == selected
            painter.setPen(QPen(QColor("#46e07a" if chosen else "#566172"), 3 if chosen else 1))
            for a, b in zip(path, path[1:]):
                painter.drawLine(*point(*world(a[0], a[1])), *point(*world(b[0], b[1])))
            if chosen and path:
                x, y = point(*world(*path[-1])); score = scores[index] if index < len(scores) else 0.0
                painter.drawText(x + 4, y, f"OUTPUT #{index + 1} score={score:.2f}")
        painter.setPen(QPen(QColor("#36a7ff"), 2)); painter.setBrush(QColor("#36a7ff"))
        for x, z in self.observation.people_xz or []:
            ix, iy = point(*world(x, z)); painter.drawEllipse(ix - 5, iy - 5, 10, 10)
            painter.drawText(ix + 7, iy, f"human {z:.2f}m")
        painter.setPen(QColor("#e5e9ef"))
        painter.drawText(10, 18, "click goal | magenta=goal | blue=human model input | green=chosen path")


class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    depth_frame_ready = pyqtSignal(QImage)
    observation_ready = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self, use_realsense: bool, camera_index: int, model_path: str,
                 confidence: float, socialwalker_checkpoint: str, remote_url: str | None = None):
        super().__init__()
        self.use_realsense = use_realsense
        self.camera_index = camera_index
        self.model_path = model_path
        self.confidence = confidence
        self.socialwalker_checkpoint = socialwalker_checkpoint
        self.remote_url = remote_url
        self.running = True
        self.previous_center: float | None = None
        self.previous_time = 0.0
        self.filtered_velocity = 0.0

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        try:
            import cv2
            import numpy as np
            from ultralytics import YOLO
        except ImportError as exc:
            self.status.emit(f"Camera dependency missing: {exc}")
            return

        try:
            model = YOLO(self.model_path)
            person_classes = [int(class_id) for class_id, name in model.names.items()
                              if str(name).strip().lower() == "person"]
            if not person_classes:
                raise RuntimeError("YOLO model has no class named 'person'")
        except Exception as exc:
            self.status.emit(f"Cannot load YOLO model: {exc}")
            return

        try:
            from socialwalker.realtime_policy import SocialWalkerPolicy
            social_policy = SocialWalkerPolicy(self.socialwalker_checkpoint)
            accuracy = ("unknown" if social_policy.best_val_acc is None
                        else f"{social_policy.best_val_acc * 100:.1f}%")
            self.status.emit(f"YOLO + SocialWalker loaded ({social_policy.device}, "
                             f"epoch {social_policy.epoch}, validation {accuracy})")
        except Exception as exc:
            self.status.emit(f"Cannot load SocialWalker model: {exc}")
            return

        pipeline = None
        capture = None
        align = None
        depth_scale = 0.001
        try:
            if self.remote_url:
                self.status.emit(f"Remote metric RGB-D perception: {self.remote_url}")
            elif self.use_realsense:
                import pyrealsense2 as rs
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                profile = pipeline.start(config)
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                align = rs.align(rs.stream.color)
                self.status.emit("RealSense RGB-D streaming")
            else:
                capture = cv2.VideoCapture(self.camera_index)
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if not capture.isOpened():
                    raise RuntimeError(f"Cannot open camera index {self.camera_index}")
                self.status.emit("RGB webcam streaming; depth unavailable")

            while self.running:
                loop_started = time.monotonic()
                depth_image = None
                if self.remote_url:
                    rgb_bytes = self._remote_get("/rgb.jpg")
                    depth_bytes = self._remote_get("/depth.png")
                    camera_status = json.loads(self._remote_get("/status").decode("utf-8"))
                    frame = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)
                    depth_image = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    depth_scale = float(camera_status.get("depth_scale", 0.001))
                    if frame is None or depth_image is None:
                        continue
                    depth_preview = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    depth_rgb = cv2.cvtColor(depth_preview, cv2.COLOR_BGR2RGB)
                    dh, dw, dc = depth_rgb.shape
                    self.depth_frame_ready.emit(QImage(depth_rgb.data, dw, dh, dc * dw,
                                                       QImage.Format.Format_RGB888).copy())
                elif pipeline:
                    frames = align.process(pipeline.wait_for_frames(1000))
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                    frame = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                else:
                    ok, frame = capture.read()
                    if not ok:
                        continue

                # This experiment intentionally uses YOLO people only. Raw depth
                # remains an independent safety input for unclassified obstacles.
                results = model.predict(frame, classes=person_classes, conf=self.confidence,
                                        imgsz=416, verbose=False)
                obstacle_distance = self._front_obstacle_depth(depth_image, depth_scale, np)
                observation = PersonObservation(
                    depth_available=depth_image is not None,
                    obstacle_distance_m=obstacle_distance,
                )
                candidates = []
                policy_detections = []
                # Obstacles are deliberately excluded from the plot and model.
                # The scalar central-depth safety gate is evaluated separately.
                obstacle_points = []
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                        confidence = float(box.conf[0].cpu())
                        distance = self._box_depth(depth_image, x1, y1, x2, y2, depth_scale, np)
                        area = max(0.0, (x2 - x1) * (y2 - y1))
                        class_id = int(box.cls[0].cpu())
                        label = str(model.names[class_id])
                        is_person = class_id in person_classes
                        if is_person:
                            candidates.append((distance, -area, x1, y1, x2, y2, confidence))
                        if is_person and distance is not None:
                            policy_detections.append({
                                "bbox_xyxy": [x1, y1, x2, y2], "depth_m": distance,
                            })
                        color = (255, 120, 30)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        distance_text = "depth N/A" if distance is None else f"{distance:.2f} m"
                        cv2.putText(frame, f"{label} {distance_text}",
                                    (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.50, color, 2)

                policy_result = social_policy.predict(policy_detections, frame.shape[1], frame.shape[0])

                if candidates:
                    # Closest valid-depth person wins; otherwise use the largest box.
                    candidates.sort(key=lambda item: (item[0] is None, item[0] or 9999.0, item[1]))
                    distance, _, x1, y1, x2, y2, confidence = candidates[0]
                    center = (x1 + x2) * 0.5
                    now = time.monotonic()
                    if self.previous_center is not None and now > self.previous_time:
                        raw_velocity = (center - self.previous_center) / (now - self.previous_time)
                        self.filtered_velocity = 0.7 * self.filtered_velocity + 0.3 * raw_velocity
                    self.previous_center, self.previous_time = center, now
                    observation = PersonObservation(
                        True, distance, center, self.filtered_velocity, confidence,
                        depth_image is not None, obstacle_distance,
                        policy_result.motion, policy_result.speed, policy_result.confidence,
                        policy_result.reason,
                        policy_result.people_xz, obstacle_points, policy_result.trajectories,
                        policy_result.scores, policy_result.selected_index,
                    )
                else:
                    self.previous_center = None
                    self.filtered_velocity = 0.0
                    observation.policy_motion = policy_result.motion
                    observation.policy_speed = policy_result.speed
                    observation.policy_confidence = policy_result.confidence
                    observation.policy_reason = policy_result.reason
                    observation.people_xz = policy_result.people_xz
                    observation.obstacles_xz = obstacle_points
                    observation.trajectories = policy_result.trajectories
                    observation.trajectory_scores = policy_result.scores
                    observation.selected_trajectory = policy_result.selected_index

                obstacle_text = "N/A" if obstacle_distance is None else f"{obstacle_distance:.2f}m"
                cv2.putText(frame, f"front depth: {obstacle_text} | SW: {policy_result.motion}/{policy_result.speed}",
                            (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 2)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(image)
                self.observation_ready.emit(observation)
                if self.remote_url:
                    remaining = (1.0 / 3.0) - (time.monotonic() - loop_started)
                    if remaining > 0:
                        self.msleep(int(remaining * 1000))
        except Exception as exc:
            self.status.emit(f"Camera stopped: {exc}")
        finally:
            if pipeline:
                pipeline.stop()
            if capture:
                capture.release()

    def _remote_get(self, path):
        request = urllib.request.Request(self.remote_url + path, headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(request, timeout=1.5) as response:
            return response.read()

    @staticmethod
    def _box_depth(depth_image, x1, y1, x2, y2, scale, np):
        if depth_image is None:
            return None
        height, width = depth_image.shape
        # Center 40% avoids background pixels around a person's silhouette.
        xa = max(0, min(width - 1, int(x1 + 0.30 * (x2 - x1))))
        xb = max(xa + 1, min(width, int(x2 - 0.30 * (x2 - x1))))
        ya = max(0, min(height - 1, int(y1 + 0.25 * (y2 - y1))))
        yb = max(ya + 1, min(height, int(y2 - 0.25 * (y2 - y1))))
        values = depth_image[ya:yb, xa:xb]
        values = values[values > 0]
        if values.size < 10:
            return None
        return float(np.median(values) * scale)

    @staticmethod
    def _front_obstacle_depth(depth_image, scale, np):
        """Conservative depth in the corridor directly in front of the base."""
        if depth_image is None:
            return None
        height, width = depth_image.shape
        region = depth_image[int(height * 0.25):int(height * 0.82),
                             int(width * 0.35):int(width * 0.65)].astype(np.float32) * scale
        valid = region[(region >= 0.15) & (region <= 6.0)]
        if valid.size < 30:
            return None
        # A percentile rejects isolated RealSense speckles but reacts before the median.
        return float(np.percentile(valid, 10.0))

class RemoteCameraWorker(QThread):
    rgb_ready = pyqtSignal(QImage)
    depth_ready = pyqtSignal(QImage)
    status = pyqtSignal(str)

    def __init__(self, host: str, port: int):
        super().__init__()
        self.base_url = f"http://{host}:{port}"
        self.running = True

    def stop(self):
        self.running = False

    def _get(self, path):
        request = urllib.request.Request(self.base_url + path, headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(request, timeout=1.0) as response:
            return response.read()

    def run(self):
        while self.running:
            started = time.monotonic()
            try:
                rgb = QImage.fromData(self._get("/rgb.jpg"), "JPG")
                depth = QImage.fromData(self._get("/depth.jpg"), "JPG")
                status = json.loads(self._get("/status").decode("utf-8"))
                if not rgb.isNull(): self.rgb_ready.emit(rgb)
                if not depth.isNull(): self.depth_ready.emit(depth)
                self.status.emit(
                    f"Jetson camera: ready={status.get('ready')} | FPS={status.get('fps')} | "
                    f"frame age={status.get('last_frame_age_ms')} ms | error={status.get('error')}"
                )
            except Exception as exc:
                self.status.emit(f"Jetson camera unavailable: {exc}")
            remaining = (1.0 / 3.0) - (time.monotonic() - started)
            if remaining > 0:
                self.msleep(int(remaining * 1000))


class RobotCarWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RobotCar SocialWalker Diagnostics")
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry() if screen else None
        if available:
            self.resize(min(1120, int(available.width() * 0.94)),
                        min(760, int(available.height() * 0.92)))
        else:
            self.resize(1120, 760)
        self.serial_port: serial.Serial | None = None
        self.tcp_socket: socket.socket | None = None
        self.camera_worker: CameraWorker | None = None
        self.remote_camera_active = False
        self.settings = QSettings("RobotCar", "ControlApp")
        self.speed_state = "6"
        self.motion_state = "S"
        self.keys_down: set[int] = set()
        self.last_observation = PersonObservation()
        self.last_detection_time = 0.0
        self.serial_rx = bytearray()
        self.firmware_verified = False
        self.connected_at = 0.0
        self.handshake_warning_shown = False
        self.pending_commands = defaultdict(deque)
        self.tx_count = 0
        self.rx_count = 0
        self.ack_count = 0
        self.heartbeat_count = 0
        self.timeout_count = 0
        self.robot_pose = [0.0, 0.0, 0.0]  # world x, world y, heading; estimated only
        self.navigation_goal = None
        self.navigation_active = False
        self.pose_trail = deque([(0.0, 0.0)], maxlen=500)
        self.last_pose_update = time.monotonic()
        self.human_test_deadline = None
        self._build_ui()
        QApplication.instance().installEventFilter(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._control_tick)
        self.timer.start(100)
        self.refresh_ports()

    def _build_ui(self):
        root = QWidget()
        self.outer = QHBoxLayout(root)
        self.outer.setContentsMargins(8, 8, 8, 8)
        self.outer.setSpacing(8)
        controls_panel = QWidget(); controls = QVBoxLayout(controls_panel)
        vision_panel = QWidget(); vision = QVBoxLayout(vision_panel)
        self.outer.addWidget(controls_panel, 1)
        self.outer.addWidget(vision_panel, 2)

        serial_box = QGroupBox("Connection: local serial or headless Jetson bridge")
        serial_row = QGridLayout(serial_box)
        self.transport_box = QComboBox(); self.transport_box.addItems(["Jetson TCP", "Local serial"])
        self.transport_box.currentTextChanged.connect(self.transport_changed)
        self.port_box = QComboBox(); self.baud_box = QComboBox(); self.baud_box.addItems(["115200", "9600"])
        self.jetson_host = QLineEdit(self.settings.value("jetson_host", "nano.local"))
        self.jetson_port = QSpinBox(); self.jetson_port.setRange(1, 65535); self.jetson_port.setValue(8765)
        refresh = QPushButton("Refresh"); refresh.clicked.connect(self.refresh_ports)
        self.connect_button = QPushButton("Connect"); self.connect_button.clicked.connect(self.toggle_serial)
        self.serial_status = QLabel("Disconnected")
        serial_row.addWidget(self.transport_box, 0, 0); serial_row.addWidget(self.jetson_host, 0, 1)
        serial_row.addWidget(self.jetson_port, 0, 2); serial_row.addWidget(self.connect_button, 0, 3)
        serial_row.addWidget(self.port_box, 1, 0); serial_row.addWidget(self.baud_box, 1, 1)
        serial_row.addWidget(refresh, 1, 2); serial_row.addWidget(self.serial_status, 2, 0, 1, 4)
        controls.addWidget(serial_box)
        self.transport_changed(self.transport_box.currentText())

        speed_box = QGroupBox("Target speed levels (firmware ramps between levels)")
        speed_grid = QGridLayout(speed_box)
        for index, (state, pwm) in enumerate(SPEED_STATES.items()):
            button = QPushButton(f"{state} -> {pwm}")
            button.clicked.connect(lambda _, s=state: self.set_speed(s))
            speed_grid.addWidget(button, index // 4, index % 4)
        controls.addWidget(speed_box)

        manual_box = QGroupBox("Manual: hold W/A/S/D; diagonals move and turn")
        grid = QGridLayout(manual_box)
        buttons = [
            ("G: forward-left", "G", 0, 0), ("F: forward", "F", 0, 1), ("I: forward-right", "I", 0, 2),
            ("L: pivot left", "L", 1, 0), ("STOP", "S", 1, 1), ("R: pivot right", "R", 1, 2),
            ("H: reverse-left", "H", 2, 0), ("B: reverse", "B", 2, 1), ("J: reverse-right", "J", 2, 2),
        ]
        for label, state, row, column in buttons:
            button = QPushButton(label)
            if state == "S": button.clicked.connect(self.stop_motion)
            else:
                button.pressed.connect(lambda s=state: self.set_motion(s))
                button.released.connect(self.stop_motion)
            grid.addWidget(button, row, column)
        controls.addWidget(manual_box)

        safety = QHBoxLayout()
        emergency = QPushButton("EMERGENCY"); emergency.clicked.connect(self.emergency)
        emergency.setStyleSheet("background:#b71c1c;color:white;font-weight:bold;padding:10px")
        reset = QPushButton("Reset E-stop"); reset.clicked.connect(lambda: self.send("X", force=True))
        safety.addWidget(emergency); safety.addWidget(reset); controls.addLayout(safety)
        self.drive_status = QLabel("Speed 6/PWM155 | STOP")
        controls.addWidget(self.drive_status)

        history_box = QGroupBox("Command history / ACK diagnostics")
        history_layout = QVBoxLayout(history_box)
        history_toolbar = QHBoxLayout()
        self.serial_counters = QLabel("CMD 0 | ACK 0 | timeout 0 | pending 0 | heartbeat 0")
        self.show_heartbeat = QCheckBox("Show heartbeat K")
        clear_history = QPushButton("Clear")
        clear_history.clicked.connect(self.clear_serial_history)
        history_toolbar.addWidget(self.serial_counters, 1)
        history_toolbar.addWidget(self.show_heartbeat)
        history_toolbar.addWidget(clear_history)
        self.serial_history = QPlainTextEdit()
        self.serial_history.setReadOnly(True)
        self.serial_history.setMaximumBlockCount(1000)
        self.serial_history.setMinimumHeight(170)
        history_layout.addLayout(history_toolbar)
        history_layout.addWidget(self.serial_history)
        controls.addWidget(history_box)
        controls.addStretch()

        camera_box = QGroupBox("RGB-D + person-only YOLO + SocialWalker policy")
        camera_grid = QGridLayout(camera_box)
        self.camera_source = QComboBox(); self.camera_source.addItems(["Jetson RGB-D stream", "Local camera"])
        self.camera_http_port = QSpinBox(); self.camera_http_port.setRange(1, 65535); self.camera_http_port.setValue(8080)
        self.realsense_check = QCheckBox("Intel RealSense RGB-D"); self.realsense_check.setChecked(True)
        self.camera_index = QSpinBox(); self.camera_index.setRange(0, 10)
        self.model_path = QLineEdit("yolo11n.pt")
        default_checkpoint = PROJECT_ROOT / "socialwalker" / "ckpt_rank.pt"
        self.socialwalker_checkpoint = QLineEdit(str(default_checkpoint))
        self.confidence = QDoubleSpinBox(); self.confidence.setRange(0.1, 0.95); self.confidence.setValue(0.45)
        self.camera_button = QPushButton("Start camera"); self.camera_button.clicked.connect(self.toggle_camera)
        self.autonomy_check = QCheckBox("ENABLE CLICK-TO-GO AUTONOMY")
        self.autonomy_check.setStyleSheet("font-weight:bold;color:#b71c1c")
        self.autonomy_check.toggled.connect(self.autonomy_toggled)
        camera_grid.addWidget(self.camera_source, 0, 0, 1, 2)
        camera_grid.addWidget(QLabel("HTTP port"), 0, 2); camera_grid.addWidget(self.camera_http_port, 0, 3)
        camera_grid.addWidget(self.realsense_check, 1, 0)
        camera_grid.addWidget(QLabel("Webcam index"), 1, 1); camera_grid.addWidget(self.camera_index, 1, 2)
        camera_grid.addWidget(QLabel("YOLO model"), 2, 0); camera_grid.addWidget(self.model_path, 2, 1, 1, 2)
        camera_grid.addWidget(QLabel("SocialWalker checkpoint"), 3, 0)
        camera_grid.addWidget(self.socialwalker_checkpoint, 3, 1, 1, 3)
        camera_grid.addWidget(QLabel("Confidence"), 4, 0); camera_grid.addWidget(self.confidence, 4, 1)
        camera_grid.addWidget(self.camera_button, 4, 2)
        camera_grid.addWidget(self.autonomy_check, 5, 0, 1, 4)
        self.navigation_status = QLabel("Click a magenta goal on the map (coordinates are meters from start)")
        reset_map = QPushButton("Reset estimated pose")
        reset_map.clicked.connect(self.reset_navigation)
        camera_grid.addWidget(self.navigation_status, 6, 0, 1, 3)
        camera_grid.addWidget(reset_map, 6, 3)
        self.human_test_button = QPushButton("Run ahead for 1 minute (human test)")
        self.human_test_button.clicked.connect(self.toggle_human_test)
        camera_grid.addWidget(self.human_test_button, 7, 0, 1, 4)
        vision.addWidget(camera_box)
        previews = QHBoxLayout()
        self.video = QLabel("RGB stopped"); self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_video = QLabel("Depth stopped"); self.depth_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for preview in (self.video, self.depth_video):
            preview.setMinimumSize(160, 120)
            preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
            preview.setStyleSheet("background:#111;color:white")
            previews.addWidget(preview, 1)
        vision.addLayout(previews, 1)
        self.perception_status = QLabel("No observation")
        self.policy_status = QLabel("Policy disabled")
        vision.addWidget(self.perception_status); vision.addWidget(self.policy_status)
        self.model_plot = ModelPlot()
        self.model_plot.goal_clicked.connect(self.set_navigation_goal)
        vision.addWidget(self.model_plot, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        self._update_layout_direction(self.width())

    def _update_layout_direction(self, width):
        """Stack the two main panels when the available window is narrow."""
        direction = (QBoxLayout.Direction.LeftToRight if width >= 1000
                     else QBoxLayout.Direction.TopToBottom)
        if self.outer.direction() != direction:
            self.outer.setDirection(direction)
            self.outer.setStretch(0, 1)
            self.outer.setStretch(1, 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "outer"):
            self._update_layout_direction(event.size().width())

    def refresh_ports(self):
        selected = self.port_box.currentText()
        self.port_box.clear(); self.port_box.addItems([p.device for p in list_ports.comports()])
        if selected: self.port_box.setCurrentText(selected)

    def transport_changed(self, transport):
        local = transport == "Local serial"
        self.port_box.setEnabled(local); self.baud_box.setEnabled(local)
        self.jetson_host.setEnabled(not local); self.jetson_port.setEnabled(not local)

    def toggle_serial(self):
        if self._connected():
            self.disconnect_serial(); return
        if self.transport_box.currentText() == "Local serial" and not self.port_box.currentText():
            QMessageBox.warning(self, "No port", "Select the HC-05 or Arduino USB serial port."); return
        try:
            if self.transport_box.currentText() == "Jetson TCP":
                self.tcp_socket = socket.create_connection(
                    (self.jetson_host.text().strip(), self.jetson_port.value()), timeout=3.0)
                self.tcp_socket.setblocking(False)
                endpoint = f"Jetson {self.jetson_host.text().strip()}:{self.jetson_port.value()}"
            else:
                self.serial_port = serial.Serial(self.port_box.currentText(), int(self.baud_box.currentText()), timeout=0)
                self.serial_port.reset_input_buffer()
                endpoint = f"{self.port_box.currentText()} @ {self.baud_box.currentText()}"
            self.firmware_verified = False; self.connected_at = time.monotonic(); self.handshake_warning_shown = False
            self.settings.setValue("jetson_host", self.jetson_host.text().strip())
            self.connect_button.setText("Disconnect"); self.serial_status.setText("Port open; waiting for firmware ACK")
            self._history(f"OPEN {endpoint}")
            self.send("X", force=True); self.set_speed(self.speed_state); self.stop_motion()
            QTimer.singleShot(700, self.retry_handshake)
        except (serial.SerialException, OSError) as exc:
            QMessageBox.critical(self, "Connection error", str(exc)); self.serial_port = None; self.tcp_socket = None

    def disconnect_serial(self):
        port, self.serial_port = self.serial_port, None
        tcp, self.tcp_socket = self.tcp_socket, None
        if port:
            try: port.write(b"S"); port.close()
            except serial.SerialException: pass
        if tcp:
            try: tcp.sendall(b"S"); tcp.close()
            except OSError: pass
        self.connect_button.setText("Connect"); self.serial_status.setText("Disconnected")
        self.firmware_verified = False; self.serial_rx.clear()
        self.pending_commands.clear(); self._history("CLOSE")
        self._update_serial_counters()

    def send(self, state: str, force=False):
        if not self._connected(): return False
        try:
            payload = state.encode("ascii")
            if self.tcp_socket: self.tcp_socket.sendall(payload)
            else: self.serial_port.write(payload)
            if state == "K":
                self.heartbeat_count += 1
            else:
                self.tx_count += 1
                self.pending_commands[state].append(time.monotonic())
            if state != "K" or self.show_heartbeat.isChecked():
                self._history(f"TX  {state}")
            self._update_serial_counters()
            return True
        except (serial.SerialException, OSError) as exc:
            self._history(f"ERROR TX {state}: {exc}")
            if force: self.disconnect_serial()
            return False

    def set_speed(self, state: str):
        self.speed_state = state; self.send(state)
        self._update_drive_status()

    def set_motion(self, state: str):
        if self.autonomy_check.isChecked(): return
        self.motion_state = state; self.send(state); self._update_drive_status()

    def stop_motion(self):
        self.motion_state = "S"; self.send("S"); self._update_drive_status()

    def emergency(self):
        self.autonomy_check.setChecked(False); self.motion_state = "S"; self.send("E", force=True)
        self._update_drive_status()

    def _update_drive_status(self):
        self.drive_status.setText(
            f"Speed {self.speed_state}/PWM{SPEED_STATES[self.speed_state]} | {MOTION_LABELS[self.motion_state]}"
        )

    def toggle_camera(self):
        if self.camera_worker:
            self.camera_worker.stop(); self.camera_worker.wait(2000); self.camera_worker = None
            self.remote_camera_active = False
            self.camera_button.setText("Start camera"); self.video.setText("RGB stopped"); self.depth_video.setText("Depth stopped")
            return
        if self.camera_source.currentText() == "Jetson RGB-D stream":
            host = self.jetson_host.text().strip()
            remote_url = f"http://{host}:{self.camera_http_port.value()}"
            self.camera_worker = CameraWorker(
                False, self.camera_index.value(), self.model_path.text().strip(),
                self.confidence.value(), self.socialwalker_checkpoint.text().strip(), remote_url)
            self.camera_worker.frame_ready.connect(self.show_frame)
            self.camera_worker.depth_frame_ready.connect(self.show_depth_frame)
            self.camera_worker.observation_ready.connect(self.handle_observation)
            self.camera_worker.status.connect(self.perception_status.setText)
            self.remote_camera_active = True
        else:
            self.camera_worker = CameraWorker(self.realsense_check.isChecked(), self.camera_index.value(),
                                              self.model_path.text().strip(), self.confidence.value(),
                                              self.socialwalker_checkpoint.text().strip())
            self.camera_worker.frame_ready.connect(self.show_frame)
            self.camera_worker.observation_ready.connect(self.handle_observation)
            self.camera_worker.status.connect(self.perception_status.setText)
        self.camera_worker.finished.connect(self.camera_finished)
        self.camera_worker.start(); self.camera_button.setText("Stop camera")

    def camera_finished(self):
        self.camera_worker = None; self.remote_camera_active = False; self.camera_button.setText("Start camera")
        if self.autonomy_check.isChecked(): self.autonomy_check.setChecked(False)

    def show_frame(self, image: QImage):
        self.video.setPixmap(QPixmap.fromImage(image).scaled(
            self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def show_depth_frame(self, image: QImage):
        self.depth_video.setPixmap(QPixmap.fromImage(image).scaled(
            self.depth_video.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def handle_observation(self, observation: PersonObservation):
        self.last_observation = observation; self.last_detection_time = time.monotonic()
        if self.human_test_deadline is not None:
            heading = self.robot_pose[2]
            self.navigation_goal = (self.robot_pose[0] - math.sin(heading) * 5.0,
                                    self.robot_pose[1] + math.cos(heading) * 5.0)
        if self.navigation_goal is not None:
            self._select_goal_path(observation)
        self.model_plot.set_observation(observation)
        self.model_plot.set_navigation(tuple(self.robot_pose), self.navigation_goal, self.pose_trail)
        if not observation.detected:
            self.perception_status.setText("No person detected")
        else:
            depth = "N/A" if observation.distance_m is None else f"{observation.distance_m:.2f} m"
            obstacle = "N/A" if observation.obstacle_distance_m is None else f"{observation.obstacle_distance_m:.2f} m"
            self.perception_status.setText(
                f"Person: depth={depth}, horizontal velocity={observation.velocity_x_px_s:.0f} px/s, "
                f"confidence={observation.confidence:.2f} | front obstacle={obstacle}")
        if self.autonomy_check.isChecked(): self.apply_person_policy(observation)

    def autonomy_toggled(self, enabled: bool):
        if enabled and not self._connected():
            QMessageBox.warning(self, "Robot disconnected", "Connect and verify the robot bridge before enabling autonomy.")
            self.autonomy_check.blockSignals(True); self.autonomy_check.setChecked(False); self.autonomy_check.blockSignals(False)
            return
        if enabled and self.navigation_goal is None:
            QMessageBox.warning(self, "No destination", "Click a destination on the 2D map first.")
            self.autonomy_check.blockSignals(True); self.autonomy_check.setChecked(False); self.autonomy_check.blockSignals(False)
            return
        if enabled and not self.camera_worker:
            QMessageBox.warning(self, "Perception required",
                                "Start the local or Jetson RGB-D perception stream before autonomous motion.")
            self.autonomy_check.blockSignals(True); self.autonomy_check.setChecked(False); self.autonomy_check.blockSignals(False)
            return
        self.navigation_active = enabled
        if not enabled:
            if self.human_test_deadline is not None:
                self.human_test_deadline = None
                self.human_test_button.setText("Run ahead for 1 minute (human test)")
            self.stop_motion(); self.policy_status.setText("Policy disabled")

    def set_navigation_goal(self, x: float, y: float):
        self.navigation_goal = (x, y)
        self.navigation_active = self.autonomy_check.isChecked()
        self.navigation_status.setText(f"Goal x={x:.2f} m, y={y:.2f} m | dead-reckoned pose")
        self.model_plot.set_navigation(tuple(self.robot_pose), self.navigation_goal, self.pose_trail)

    def reset_navigation(self):
        self.human_test_deadline = None
        self.human_test_button.setText("Run ahead for 1 minute (human test)")
        self.autonomy_check.setChecked(False)
        self.robot_pose[:] = [0.0, 0.0, 0.0]
        self.navigation_goal = None
        self.navigation_active = False
        self.pose_trail.clear(); self.pose_trail.append((0.0, 0.0))
        self.last_pose_update = time.monotonic()
        self.navigation_status.setText("Pose reset. Click a new goal on the map.")
        self.model_plot.set_navigation(tuple(self.robot_pose), None, self.pose_trail)

    def toggle_human_test(self):
        if self.human_test_deadline is not None:
            self._finish_human_test("1-minute human test cancelled")
            return
        if not self._connected():
            QMessageBox.warning(self, "Robot disconnected", "Connect the robot bridge before starting the test.")
            return
        if not self.camera_worker:
            QMessageBox.warning(self, "Camera stopped", "Start the RGB-D camera before starting the test.")
            return
        heading = self.robot_pose[2]
        self.navigation_goal = (self.robot_pose[0] - math.sin(heading) * 5.0,
                                self.robot_pose[1] + math.cos(heading) * 5.0)
        self.human_test_deadline = time.monotonic() + 60.0
        self.human_test_button.setText("Cancel 1-minute test")
        self.navigation_status.setText("Human influence test: 60 s remaining")
        self.autonomy_check.setChecked(True)
        if not self.autonomy_check.isChecked():
            self.human_test_deadline = None
            self.human_test_button.setText("Run ahead for 1 minute (human test)")

    def _finish_human_test(self, message):
        self.human_test_deadline = None
        self.navigation_active = False
        self.human_test_button.setText("Run ahead for 1 minute (human test)")
        self.autonomy_check.blockSignals(True)
        self.autonomy_check.setChecked(False)
        self.autonomy_check.blockSignals(False)
        self._apply_auto_command("0", "S", message)
        self.navigation_status.setText(message)

    def _select_goal_path(self, obs: PersonObservation):
        """Fuse goal heading with SocialWalker scores from human inputs only."""
        paths, scores = obs.trajectories or [], obs.trajectory_scores or []
        if not paths or len(scores) != len(paths):
            obs.policy_motion, obs.policy_speed = "S", "0"
            obs.policy_reason = "Waiting for SocialWalker history"
            obs.selected_trajectory = -1
            return
        dx = self.navigation_goal[0] - self.robot_pose[0]
        dy = self.navigation_goal[1] - self.robot_pose[1]
        heading = self.robot_pose[2]
        local_x = math.cos(heading) * dx + math.sin(heading) * dy
        local_z = -math.sin(heading) * dx + math.cos(heading) * dy
        target_angle = math.atan2(local_x, max(0.05, local_z))
        low, high = min(scores), max(scores)
        span = max(1e-6, high - low)
        best_index, best_cost = -1, float("inf")
        for index, path in enumerate(paths):
            end_x, end_z = path[-1]
            angle_error = abs(math.atan2(math.sin(math.atan2(end_x, end_z) - target_angle),
                                         math.cos(math.atan2(end_x, end_z) - target_angle)))
            social_cost = (high - scores[index]) / span
            cost = 2.0 * angle_error + 0.65 * social_cost
            if cost < best_cost:
                best_index, best_cost = index, cost
        obs.selected_trajectory = best_index
        end_x = paths[best_index][-1][0]
        obs.policy_motion = "G" if end_x < -0.8 else "I" if end_x > 0.8 else "F"
        obs.policy_speed = "4" if obs.detected or (obs.obstacle_distance_m or 9) < 1.5 else "6"
        obs.policy_reason = f"Goal planner path {best_index + 1}/{len(paths)}"

    def apply_person_policy(self, obs: PersonObservation):
        if not self.navigation_active or self.navigation_goal is None:
            self._apply_auto_command("0", "S", "STOP: no active navigation goal")
        elif math.hypot(self.navigation_goal[0] - self.robot_pose[0],
                        self.navigation_goal[1] - self.robot_pose[1]) < 0.35:
            self.navigation_active = False
            self._apply_auto_command("0", "S", "ARRIVED: within 0.35 m (estimated)")
            self.navigation_status.setText("Goal reached by dead reckoning; autonomy stopped")
        elif not obs.depth_available:
            self._apply_auto_command("0", "S", "STOP: no depth")
        elif obs.obstacle_distance_m is None:
            self._apply_auto_command("0", "S", "STOP: invalid front depth")
        elif obs.obstacle_distance_m < 0.65:
            self._apply_auto_command("0", "S", f"STOP: obstacle {obs.obstacle_distance_m:.2f} m")
        elif obs.detected and obs.distance_m is None:
            self._apply_auto_command("0", "S", "STOP: person depth invalid")
        elif not obs.detected:
            self._apply_auto_command(
                obs.policy_speed, obs.policy_motion,
                f"{obs.policy_reason} (no human input), confidence={obs.policy_confidence:.2f}",
            )
        elif obs.distance_m < 0.85:
            self._apply_auto_command("0", "S", "STOP: person < 0.85 m")
        else:
            speed = "4" if obs.distance_m < 1.50 else obs.policy_speed
            self._apply_auto_command(
                speed, obs.policy_motion,
                f"{obs.policy_reason}, confidence={obs.policy_confidence:.2f}",
            )

    def _apply_auto_command(self, speed: str, motion: str, reason: str):
        if speed != self.speed_state: self.set_speed(speed)
        if motion != self.motion_state:
            self.motion_state = motion; self.send(motion); self._update_drive_status()
        self.policy_status.setText(reason)

    def _control_tick(self):
        self._update_estimated_pose()
        if self.human_test_deadline is not None:
            remaining = self.human_test_deadline - time.monotonic()
            if remaining <= 0:
                self._finish_human_test("1-minute human test complete: STOP")
            else:
                self.navigation_status.setText(f"Human influence test: {remaining:.0f} s remaining")
        if self._connected():
            self.send("K")
            self._read_serial()
            self._expire_pending_commands()
            if (not self.firmware_verified and not self.handshake_warning_shown and
                    time.monotonic() - self.connected_at > 1.5):
                self.handshake_warning_shown = True
                self.serial_status.setText("Port open, but NO firmware ACK")
                QMessageBox.warning(
                    self, "No firmware response",
                    "The connection opened, but the RobotCar firmware did not ACK. For Jetson TCP, check the bridge "
                    "service and /dev/robot_base. For HC-05, select its outgoing COM port at 9600 baud.",
                )
        # Loss of camera/detection messages must stop autonomous motion.
        if self.autonomy_check.isChecked() and time.monotonic() - self.last_detection_time > 0.5:
            self._apply_auto_command("0", "S", "STOP: stale camera")

    def _update_estimated_pose(self):
        now = time.monotonic()
        dt = min(0.25, max(0.0, now - self.last_pose_update))
        self.last_pose_update = now
        if not (self.navigation_active and self._connected()):
            return
        linear_by_speed = {"0": 0.0, "4": 0.20, "6": 0.32, "7": 0.38,
                           "8": 0.43, "9": 0.50, "q": 0.56}
        linear = linear_by_speed.get(self.speed_state, 0.0)
        angular = {"G": 0.55, "I": -0.55, "L": 1.0, "R": -1.0}.get(self.motion_state, 0.0)
        if self.motion_state in ("S", "L", "R"):
            linear = 0.0
        elif self.motion_state in ("B", "H", "J"):
            linear = -linear
        self.robot_pose[2] += angular * dt
        self.robot_pose[0] -= math.sin(self.robot_pose[2]) * linear * dt
        self.robot_pose[1] += math.cos(self.robot_pose[2]) * linear * dt
        if math.hypot(self.robot_pose[0] - self.pose_trail[-1][0],
                      self.robot_pose[1] - self.pose_trail[-1][1]) > 0.03:
            self.pose_trail.append((self.robot_pose[0], self.robot_pose[1]))
        self.model_plot.set_navigation(tuple(self.robot_pose), self.navigation_goal, self.pose_trail)

    def _read_serial(self):
        try:
            if self.tcp_socket:
                try:
                    chunk = self.tcp_socket.recv(4096)
                    if not chunk:
                        self._history("ERROR Jetson bridge disconnected")
                        self.disconnect_serial(); return
                    self.serial_rx.extend(chunk)
                except BlockingIOError:
                    pass
            else:
                waiting = self.serial_port.in_waiting
                if waiting: self.serial_rx.extend(self.serial_port.read(waiting))
            while b"\n" in self.serial_rx:
                raw, _, remainder = self.serial_rx.partition(b"\n")
                self.serial_rx = bytearray(remainder)
                line = raw.decode("ascii", errors="replace").strip()
                if not line:
                    continue
                self.rx_count += 1
                latency_text = ""
                if line.startswith("ACK,"):
                    parts = line.split(',')
                    command = parts[1] if len(parts) > 1 else "?"
                    self.ack_count += 1
                    if self.pending_commands[command]:
                        latency_ms = (time.monotonic() - self.pending_commands[command].popleft()) * 1000.0
                        latency_text = f" | RTT {latency_ms:.1f} ms"
                    else:
                        latency_text = " | unmatched ACK"
                self._history(f"RX  {line}{latency_text}")
                if line.startswith("ACK,") or line.startswith("READY,"):
                    self.firmware_verified = True
                    self.serial_status.setText(f"Firmware verified | {line}")
                elif line == "BRIDGE_READY":
                    self.serial_status.setText("Jetson bridge connected; waiting for Arduino ACK")
                elif line == "BUSY":
                    self.serial_status.setText("Jetson bridge busy")
                self._update_serial_counters()
        except (serial.SerialException, OSError):
            self.disconnect_serial()

    def _connected(self):
        return bool(self.tcp_socket or (self.serial_port and self.serial_port.is_open))

    def retry_handshake(self):
        if self._connected() and not self.firmware_verified:
            self.send("X")
            self.set_speed(self.speed_state)
            self.stop_motion()

    def _history(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        milliseconds = int((time.time() % 1) * 1000)
        self.serial_history.appendPlainText(f"{timestamp}.{milliseconds:03d}  {message}")

    def _update_serial_counters(self):
        pending = sum(len(items) for items in self.pending_commands.values())
        self.serial_counters.setText(
            f"CMD {self.tx_count} | ACK {self.ack_count} | timeout {self.timeout_count} | "
            f"pending {pending} | heartbeat {self.heartbeat_count}"
        )

    def _expire_pending_commands(self):
        now = time.monotonic()
        for command, timestamps in self.pending_commands.items():
            while timestamps and now - timestamps[0] > 2.0:
                timestamps.popleft()
                self.timeout_count += 1
                self._history(f"TIMEOUT {command} (no ACK after 2000 ms)")
        self._update_serial_counters()

    def clear_serial_history(self):
        self.serial_history.clear()
        self.pending_commands.clear()
        self.tx_count = self.rx_count = self.ack_count = self.heartbeat_count = self.timeout_count = 0
        self._update_serial_counters()

    def eventFilter(self, watched, event):
        if event.type() not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return super().eventFilter(watched, event)
        if self.model_path.hasFocus():
            return super().eventFilter(watched, event)
        relevant = (Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D, Qt.Key.Key_Space)
        if event.key() not in relevant:
            return super().eventFilter(watched, event)
        if event.type() == QEvent.Type.KeyPress:
            self.keyPressEvent(event)
        else:
            self.keyReleaseEvent(event)
        return True

    def keyPressEvent(self, event: QKeyEvent):
        if event.isAutoRepeat() or self.autonomy_check.isChecked(): return
        if event.key() == Qt.Key.Key_Space: self.stop_motion(); return
        if event.key() in (Qt.Key.Key_W, Qt.Key.Key_A, Qt.Key.Key_S, Qt.Key.Key_D):
            self.keys_down.add(event.key()); self._apply_keys()
        else: super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.isAutoRepeat(): return
        self.keys_down.discard(event.key()); self._apply_keys()

    def _apply_keys(self):
        w, a, s, d = (Qt.Key.Key_W in self.keys_down, Qt.Key.Key_A in self.keys_down,
                      Qt.Key.Key_S in self.keys_down, Qt.Key.Key_D in self.keys_down)
        state = "G" if w and a else "I" if w and d else "H" if s and a else "J" if s and d else \
                "F" if w else "B" if s else "L" if a else "R" if d else "S"
        self.motion_state = state; self.send(state); self._update_drive_status()

    def closeEvent(self, event):
        self.autonomy_check.setChecked(False)
        if self.camera_worker: self.camera_worker.stop(); self.camera_worker.wait(2000)
        self.disconnect_serial(); event.accept()


def main():
    app = QApplication(sys.argv); window = RobotCarWindow(); window.show(); return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
