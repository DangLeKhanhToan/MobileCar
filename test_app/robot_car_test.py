"""Simple RobotCar manual and RGB-D person-avoidance test application."""

import sys
import time
import socket
import json
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass

import serial
from serial.tools import list_ports
from PyQt6.QtCore import QEvent, QSettings, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)


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


class CameraWorker(QThread):
    frame_ready = pyqtSignal(QImage)
    observation_ready = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self, use_realsense: bool, camera_index: int, model_path: str, confidence: float):
        super().__init__()
        self.use_realsense = use_realsense
        self.camera_index = camera_index
        self.model_path = model_path
        self.confidence = confidence
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

        pipeline = None
        capture = None
        align = None
        depth_scale = 0.001
        try:
            if self.use_realsense:
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
                depth_image = None
                if pipeline:
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

                results = model.predict(frame, classes=person_classes, conf=self.confidence, imgsz=416, verbose=False)
                observation = PersonObservation(depth_available=depth_image is not None)
                candidates = []
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                        confidence = float(box.conf[0].cpu())
                        distance = self._box_depth(depth_image, x1, y1, x2, y2, depth_scale, np)
                        area = max(0.0, (x2 - x1) * (y2 - y1))
                        candidates.append((distance, -area, x1, y1, x2, y2, confidence))

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
                        True, distance, center, self.filtered_velocity, confidence, depth_image is not None
                    )
                    color = (0, 0, 255) if distance is not None and distance < 1.0 else (0, 200, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    distance_text = "depth N/A" if distance is None else f"{distance:.2f} m"
                    cv2.putText(frame, f"person {distance_text} vx={self.filtered_velocity:.0f}",
                                (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                else:
                    self.previous_center = None
                    self.filtered_velocity = 0.0

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb.shape
                image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
                self.frame_ready.emit(image)
                self.observation_ready.emit(observation)
        except Exception as exc:
            self.status.emit(f"Camera stopped: {exc}")
        finally:
            if pipeline:
                pipeline.stop()
            if capture:
                capture.release()

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
            remaining = 0.2 - (time.monotonic() - started)
            if remaining > 0:
                self.msleep(int(remaining * 1000))


class RobotCarWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RobotCar Simple Drive + Person Avoidance")
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
        self._build_ui()
        QApplication.instance().installEventFilter(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._control_tick)
        self.timer.start(100)
        self.refresh_ports()

    def _build_ui(self):
        root = QWidget(); outer = QHBoxLayout(root)
        controls = QVBoxLayout(); outer.addLayout(controls, 1)
        vision = QVBoxLayout(); outer.addLayout(vision, 2)

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

        camera_box = QGroupBox("RGB-D + YOLO person policy")
        camera_grid = QGridLayout(camera_box)
        self.camera_source = QComboBox(); self.camera_source.addItems(["Jetson RGB-D stream", "Local camera"])
        self.camera_http_port = QSpinBox(); self.camera_http_port.setRange(1, 65535); self.camera_http_port.setValue(8080)
        self.realsense_check = QCheckBox("Intel RealSense RGB-D"); self.realsense_check.setChecked(True)
        self.camera_index = QSpinBox(); self.camera_index.setRange(0, 10)
        self.model_path = QLineEdit("yolo11n.pt")
        self.confidence = QDoubleSpinBox(); self.confidence.setRange(0.1, 0.95); self.confidence.setValue(0.45)
        self.camera_button = QPushButton("Start camera"); self.camera_button.clicked.connect(self.toggle_camera)
        self.autonomy_check = QCheckBox("ENABLE AUTO PERSON AVOIDANCE")
        self.autonomy_check.setStyleSheet("font-weight:bold;color:#b71c1c")
        self.autonomy_check.toggled.connect(self.autonomy_toggled)
        camera_grid.addWidget(self.camera_source, 0, 0, 1, 2)
        camera_grid.addWidget(QLabel("HTTP port"), 0, 2); camera_grid.addWidget(self.camera_http_port, 0, 3)
        camera_grid.addWidget(self.realsense_check, 1, 0)
        camera_grid.addWidget(QLabel("Webcam index"), 1, 1); camera_grid.addWidget(self.camera_index, 1, 2)
        camera_grid.addWidget(QLabel("YOLO model"), 2, 0); camera_grid.addWidget(self.model_path, 2, 1, 1, 2)
        camera_grid.addWidget(QLabel("Confidence"), 3, 0); camera_grid.addWidget(self.confidence, 3, 1)
        camera_grid.addWidget(self.camera_button, 3, 2)
        camera_grid.addWidget(self.autonomy_check, 4, 0, 1, 4)
        vision.addWidget(camera_box)
        previews = QHBoxLayout()
        self.video = QLabel("RGB stopped"); self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_video = QLabel("Depth stopped"); self.depth_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for preview in (self.video, self.depth_video):
            preview.setMinimumSize(320, 240); preview.setStyleSheet("background:#111;color:white")
            previews.addWidget(preview, 1)
        vision.addLayout(previews, 1)
        self.perception_status = QLabel("No observation")
        self.policy_status = QLabel("Policy disabled")
        vision.addWidget(self.perception_status); vision.addWidget(self.policy_status)
        self.setCentralWidget(root)

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
            self.camera_worker = RemoteCameraWorker(host, self.camera_http_port.value())
            self.camera_worker.rgb_ready.connect(self.show_frame)
            self.camera_worker.depth_ready.connect(self.show_depth_frame)
            self.camera_worker.status.connect(self.perception_status.setText)
            self.remote_camera_active = True
        else:
            self.camera_worker = CameraWorker(self.realsense_check.isChecked(), self.camera_index.value(),
                                              self.model_path.text().strip(), self.confidence.value())
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
        if not observation.detected:
            self.perception_status.setText("No person detected")
        else:
            depth = "N/A" if observation.distance_m is None else f"{observation.distance_m:.2f} m"
            self.perception_status.setText(
                f"Person: depth={depth}, horizontal velocity={observation.velocity_x_px_s:.0f} px/s, "
                f"confidence={observation.confidence:.2f}")
        if self.autonomy_check.isChecked(): self.apply_person_policy(observation)

    def autonomy_toggled(self, enabled: bool):
        if enabled and (not self.camera_worker or self.remote_camera_active or not self.realsense_check.isChecked()):
            QMessageBox.warning(self, "Local perception required",
                                "Remote video is debug-only. Start local RealSense perception before autonomous motion.")
            self.autonomy_check.blockSignals(True); self.autonomy_check.setChecked(False); self.autonomy_check.blockSignals(False)
            return
        if not enabled:
            self.stop_motion(); self.policy_status.setText("Policy disabled")

    def apply_person_policy(self, obs: PersonObservation):
        if not obs.depth_available:
            self._apply_auto_command("0", "S", "STOP: no depth")
        elif obs.detected and obs.distance_m is None:
            self._apply_auto_command("0", "S", "STOP: person depth invalid")
        elif not obs.detected:
            self._apply_auto_command("6", "F", "No person: cruise")
        elif obs.distance_m < 0.75:
            self._apply_auto_command("0", "S", "STOP: person < 0.75 m")
        else:
            if obs.distance_m < 1.20: speed = "4"
            elif obs.distance_m < 1.80: speed = "6"
            elif obs.distance_m < 2.50: speed = "7"
            else: speed = "8"
            if obs.distance_m < 2.50 and obs.velocity_x_px_s > 35:
                motion, reason = "G", "Person moving right: evade left"
            elif obs.distance_m < 2.50 and obs.velocity_x_px_s < -35:
                motion, reason = "I", "Person moving left: evade right"
            else:
                motion, reason = "F", "Person slow/stationary: reduced forward speed"
            self._apply_auto_command(speed, motion, reason)

    def _apply_auto_command(self, speed: str, motion: str, reason: str):
        if speed != self.speed_state: self.set_speed(speed)
        if motion != self.motion_state:
            self.motion_state = motion; self.send(motion); self._update_drive_status()
        self.policy_status.setText(reason)

    def _control_tick(self):
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
