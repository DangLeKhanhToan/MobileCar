#!/usr/bin/env python3
"""RealSense RGB/depth debug server for the Jetson.

HTTP endpoints:
  /rgb.jpg    latest aligned RGB frame
  /depth.jpg  latest colorized aligned depth frame
  /status     JSON camera health/metrics
  /healthz    200 when frames are fresh, otherwise 503
"""

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import signal
import threading
import time


LOG = logging.getLogger("robot-camera")


class CameraState:
    def __init__(self):
        self.lock = threading.Lock()
        self.rgb_jpeg = None
        self.depth_jpeg = None
        self.status = {
            "type": "camera_status", "ready": False, "frame": 0,
            "fps": 0.0, "last_frame_age_ms": None, "error": "starting",
        }
        self.last_frame_time = 0.0

    def update(self, rgb_jpeg, depth_jpeg, frame, fps):
        now = time.monotonic()
        with self.lock:
            self.rgb_jpeg = rgb_jpeg
            self.depth_jpeg = depth_jpeg
            self.last_frame_time = now
            self.status.update(ready=True, frame=frame, fps=round(fps, 1), error=None)

    def fail(self, message):
        with self.lock:
            self.status.update(ready=False, error=str(message))

    def snapshot(self):
        with self.lock:
            status = dict(self.status)
            age = None if not self.last_frame_time else (time.monotonic() - self.last_frame_time) * 1000
            status["last_frame_age_ms"] = None if age is None else round(age, 1)
            status["ready"] = bool(status["ready"] and age is not None and age < 1000)
            return self.rgb_jpeg, self.depth_jpeg, status


class RealSenseWorker(threading.Thread):
    def __init__(self, state, width, height, fps, jpeg_quality, stop_event):
        super().__init__(name="realsense", daemon=True)
        self.state = state
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.stop_event = stop_event

    def run(self):
        pipeline = None
        try:
            import cv2
            import numpy as np
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            pipeline.start(config)
            align = rs.align(rs.stream.color)
            colorizer = rs.colorizer()
            encode_args = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            count = 0
            window_start = time.monotonic()
            measured_fps = 0.0
            LOG.info("RealSense started at %dx%d@%d", self.width, self.height, self.fps)

            while not self.stop_event.is_set():
                frames = align.process(pipeline.wait_for_frames(1000))
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color = np.asanyarray(color_frame.get_data())
                depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                ok_rgb, rgb_jpeg = cv2.imencode(".jpg", color, encode_args)
                ok_depth, depth_jpeg = cv2.imencode(".jpg", depth_color, encode_args)
                if not ok_rgb or not ok_depth:
                    continue
                count += 1
                elapsed = time.monotonic() - window_start
                if elapsed >= 1.0:
                    measured_fps = count / elapsed
                    count = 0
                    window_start = time.monotonic()
                self.state.update(rgb_jpeg.tobytes(), depth_jpeg.tobytes(),
                                  self.state.status["frame"] + 1, measured_fps)
        except Exception as exc:
            LOG.exception("Camera stopped")
            self.state.fail(exc)
        finally:
            if pipeline is not None:
                pipeline.stop()


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            rgb, depth, status = state.snapshot()
            if self.path == "/status":
                self._send(200, "application/json", json.dumps(status).encode("utf-8"))
            elif self.path == "/healthz":
                self._send(200 if status["ready"] else 503, "application/json",
                           json.dumps(status).encode("utf-8"))
            elif self.path == "/rgb.jpg":
                self._image(rgb)
            elif self.path == "/depth.jpg":
                self._image(depth)
            else:
                self._send(404, "text/plain", b"not found\n")

        def _image(self, payload):
            if payload is None:
                self._send(503, "text/plain", b"camera frame unavailable\n")
            else:
                self._send(200, "image/jpeg", payload)

        def _send(self, code, content_type, payload):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt, *args):
            LOG.debug("HTTP %s - %s", self.client_address[0], fmt % args)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    state = CameraState()
    stop_event = threading.Event()
    worker = RealSenseWorker(state, args.width, args.height, args.fps, args.jpeg_quality, stop_event)
    server = ThreadingHTTPServer((args.listen, args.port), make_handler(state))

    def stop(_signum=None, _frame=None):
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    worker.start()
    LOG.info("Camera debug server listening on %s:%d", args.listen, args.port)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        stop_event.set()
        worker.join(timeout=3)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
