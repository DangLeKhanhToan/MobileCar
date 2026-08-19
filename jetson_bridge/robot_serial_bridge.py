"""Headless TCP-to-Arduino bridge for Jetson Orin Nano."""

import argparse
import selectors
import signal
import socket
import sys
import time

import serial

ALLOWED_COMMANDS = set(b"046789qFBLRGIHJSKEX")
HEARTBEAT = ord("K")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="/dev/robot_base")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--status-interval", type=float, default=2.0,
                        help="seconds between BRIDGE_STATUS messages; 0 disables")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arduino = serial.Serial(args.serial, args.baud, timeout=0, write_timeout=0.2)
    arduino.reset_input_buffer()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.listen, args.port))
    server.listen(1)
    server.setblocking(False)
    selector = selectors.DefaultSelector()
    selector.register(server, selectors.EVENT_READ, "server")
    client = None
    serial_buffer = bytearray()
    running = True
    started_at = time.monotonic()
    next_status = started_at + max(0.1, args.status_interval)
    client_rx_count = 0
    serial_line_count = 0

    def stop(_signum=None, _frame=None):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"Bridge listening on {args.listen}:{args.port}, Arduino {args.serial}@{args.baud}", flush=True)

    try:
        while running:
            for key, _ in selector.select(timeout=0.02):
                if key.data == "server":
                    new_client, address = server.accept()
                    new_client.setblocking(False)
                    if client is not None:
                        new_client.sendall(b"BUSY\n")
                        new_client.close()
                        continue
                    client = new_client
                    selector.register(client, selectors.EVENT_READ, "client")
                    client.sendall(b"BRIDGE_READY\n")
                    print(f"Laptop connected: {address}", flush=True)
                else:
                    try:
                        payload = client.recv(256)
                    except ConnectionError:
                        payload = b""
                    if not payload:
                        selector.unregister(client)
                        client.close()
                        client = None
                        arduino.write(b"S")
                        print("Laptop disconnected; STOP sent", flush=True)
                    else:
                        valid = bytes(byte for byte in payload if byte in ALLOWED_COMMANDS)
                        if valid:
                            arduino.write(valid)
                            client_rx_count += len(valid)
                            commands = bytes(byte for byte in valid if byte != HEARTBEAT)
                            if commands:
                                print(f"Commands -> Arduino: {commands.decode('ascii')}", flush=True)

            waiting = arduino.in_waiting
            if waiting:
                serial_buffer.extend(arduino.read(waiting))
            while b"\n" in serial_buffer:
                line, _, remainder = serial_buffer.partition(b"\n")
                serial_buffer = bytearray(remainder)
                line = line.rstrip(b"\r")
                serial_line_count += 1
                if client and line:
                    try:
                        client.sendall(line + b"\n")
                    except ConnectionError:
                        pass

            now = time.monotonic()
            if client and args.status_interval > 0 and now >= next_status:
                status = (
                    f"BRIDGE_STATUS,uptime_s={int(now - started_at)},"
                    f"commands={client_rx_count},arduino_lines={serial_line_count}\n"
                ).encode("ascii")
                try:
                    client.sendall(status)
                except ConnectionError:
                    pass
                next_status = now + args.status_interval
    finally:
        try:
            arduino.write(b"S")
        except serial.SerialException:
            pass
        if client:
            client.close()
        server.close()
        arduino.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
