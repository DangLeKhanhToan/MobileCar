#!/usr/bin/env python3
"""Safe TCP bridge/firmware diagnostic; motion requires --allow-motion."""

import argparse
import socket
import time


def read_line(sock: socket.socket, timeout: float = 2.0) -> str:
    sock.settimeout(timeout)
    data = bytearray()
    while not data.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("bridge closed the connection")
        data.extend(chunk)
    return data.decode("ascii", errors="replace").strip()


def wait_for(sock: socket.socket, prefix: str, timeout: float = 3.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = read_line(sock, max(0.05, deadline - time.monotonic()))
        print(f"RX  {line}")
        if line.startswith(prefix):
            return line
    raise TimeoutError(f"no response beginning with {prefix!r}")


def send_expect(sock: socket.socket, command: str) -> str:
    print(f"TX  {command}")
    sock.sendall(command.encode("ascii"))
    return wait_for(sock, f"ACK,{command},")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", help="Jetson IP address or hostname")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-motion", action="store_true",
                        help="run a raised-wheel forward/watchdog test")
    args = parser.parse_args()

    with socket.create_connection((args.host, args.port), timeout=3.0) as sock:
        wait_for(sock, "BRIDGE_READY")
        # The default test only latches emergency stop, so an accidental floor
        # test cannot move the robot.
        send_expect(sock, "E")
        print("PASS: Wi-Fi/TCP, Jetson bridge, USB serial, and firmware ACK path work.")
        if not args.allow_motion:
            print("Robot remains E-stopped. Use --allow-motion only with wheels raised.")
            return 0

        send_expect(sock, "X")
        send_expect(sock, "4")
        send_expect(sock, "F")
        print("No heartbeat for 0.8 s; firmware watchdog must stop motors within 0.5 s.")
        time.sleep(0.8)
        send_expect(sock, "E")
        print("PASS if the raised wheels stopped automatically before E was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
