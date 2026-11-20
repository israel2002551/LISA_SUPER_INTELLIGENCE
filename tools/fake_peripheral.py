"""Fake UART peripheral for bring-up: lets you test the edge node's UART stack
with nothing but a USB-UART adapter.

Wiring (3.3 V logic only!):
    adapter TX  -> ESP32-S3 GPIO18 (EDGE_UART_RX_PIN)
    adapter RX  <- ESP32-S3 GPIO17 (EDGE_UART_TX_PIN)
    adapter GND -- ESP32-S3 GND

Modes:
    smart     answers IDENTIFY with a thermostat manifest, accepts UPGRADE (with the
              1-second PING-or-revert rule), answers PING, emits temperature events,
              acks actions and sends feedback by its own policy (default)
    raw-nmea  a "legacy" GPS that ignores input and streams NMEA sentences
    raw-bin   streams Modbus RTU frames (binary, with CRC)

    python tools/fake_peripheral.py COM5                      smart device at 115200
    python tools/fake_peripheral.py COM5 --baud 9600 --sync   send 0x55 sync bytes before each line
    python tools/fake_peripheral.py COM5 --mode raw-nmea --baud 9600

Needs pyserial (pip install pyserial, or PlatformIO's Python).
"""

from __future__ import annotations

import argparse
import json
import random
import time

THERMO_MANIFEST = {
    "id": "thermo-01", "type": "thermostat", "fw": "1.0", "emits": ["temp", "hum"],
    "accepts": [{"cmd": "set_fan", "args": {"level": "int:0..3"}}, {"cmd": "beep", "args": {}}],
    "needs": ["alerts"], "acks": True, "bus": {"max_baud": 2000000, "links": ["uart"]},
    "rules": [
        {"when": {"ev": "temp", "gt": 30}, "do": {"cmd": "set_fan", "args": {"level": 2}}},
        {"when": {"ev": "temp", "lt": 15}, "do": {"cmd": "set_fan", "args": {"level": 0}}},
    ],
}


def compact(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


class SmartThermostat:
    """Device-side protocol logic, independent of the serial port (testable)."""

    def __init__(self, baud: int, rng: random.Random | None = None) -> None:
        self.baud = baud
        self.rng = rng or random.Random()
        self.temp = 22.0
        self.last_temp = self.temp
        self.pending_revert: tuple[float, int] | None = None   # (deadline, old baud) after UPGRADE
        self.log: list[str] = []

    def expected_action(self, temp: float) -> dict | None:
        if temp > 30:
            return {"cmd": "set_fan", "args": {"level": 2}}
        if temp < 15:
            return {"cmd": "set_fan", "args": {"level": 0}}
        return None

    def on_line(self, line: str, now: float) -> tuple[list[str], int | None]:
        """Returns (lines to send, new baud to switch to after sending them or None)."""
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return [], None              # garbage (e.g. IDENTIFY sent at another baud)
        if not isinstance(msg, dict):
            return [], None
        cmd = msg.get("cmd")
        if cmd == "IDENTIFY":
            return [compact(THERMO_MANIFEST)], None
        if cmd == "UPGRADE":
            baud = int(msg.get("baud", 0))
            if msg.get("link") != "uart" or not 9600 <= baud <= THERMO_MANIFEST["bus"]["max_baud"]:
                return [compact({"ack": "UPGRADE", "ok": False})], None
            self.pending_revert = (now + 1.0, self.baud)
            return [compact({"ack": "UPGRADE", "ok": True})], baud
        if cmd == "PING":
            self.pending_revert = None   # the upgraded link works
            return [compact({"pong": msg.get("n")})], None
        if "id" in msg and cmd:
            spec = {c["cmd"]: c["args"] for c in THERMO_MANIFEST["accepts"]}
            if cmd not in spec:
                return [compact({"ack": msg["id"], "ok": False, "err": "unknown cmd"})], None
            out = [compact({"ack": msg["id"], "ok": True})]
            want = self.expected_action(self.last_temp)
            got = {"cmd": cmd, "args": msg.get("args") or {}}
            out.append(compact({"fb": "good" if got == want else "bad", "id": msg["id"]}))
            self.log.append(f"action {got} for {self.last_temp} C -> {'good' if got == want else 'bad'}")
            return out, None
        return [], None

    def tick(self, now: float) -> tuple[list[str], int | None]:
        """Periodic work: revert a failed upgrade, emit a sensor event."""
        if self.pending_revert and now > self.pending_revert[0]:
            old = self.pending_revert[1]
            self.pending_revert = None
            return [], old
        self.temp = max(5.0, min(40.0, self.temp + self.rng.uniform(-3, 3)))
        self.last_temp = round(self.temp, 1)
        if self.rng.random() < 0.2:
            return [compact({"ev": "hum", "v": round(self.rng.uniform(30, 70), 1)})], None
        return [compact({"ev": "temp", "v": self.last_temp})], None


def nmea_sentence(rng: random.Random) -> str:
    body = f"GPGGA,{time.strftime('%H%M%S')},4807.{rng.randint(0, 999):03d},N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"${body}*{cs:02X}"


def modbus_frame(rng: random.Random) -> bytes:
    frame = bytes([1, 3, 4]) + rng.randint(0, 65535).to_bytes(2, "big") + rng.randint(0, 65535).to_bytes(2, "big")
    crc = 0xFFFF
    for b in frame:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return frame + bytes([crc & 0xFF, crc >> 8])


def main() -> None:
    import serial

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("port")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--mode", choices=["smart", "raw-nmea", "raw-bin"], default="smart")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between events")
    p.add_argument("--sync", action="store_true", help="send 0x55 x8 before each line (auto-baud preamble)")
    args = p.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.02)
    rng = random.Random()
    dev = SmartThermostat(args.baud, rng)
    buf = b""
    next_tick = time.monotonic() + args.interval
    print(f"{args.mode} device on {args.port} at {args.baud} baud. Ctrl+C to stop.")

    def send(lines: list[str]) -> None:
        for line in lines:
            if args.sync:
                ser.write(b"\x55" * 8)
            ser.write(line.encode() + b"\n")
            print(f"  -> {line}")

    def switch(baud: int | None) -> None:
        if baud:
            ser.flush()
            time.sleep(0.01)
            ser.baudrate = baud
            dev.baud = baud
            print(f"  [baud now {baud}]")

    while True:
        now = time.monotonic()
        data = ser.read(256)
        if args.mode == "smart" and data:
            buf += data
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").strip()
                if line.startswith("{"):
                    print(f"<-   {line}")
                out, baud = dev.on_line(line, now)
                send(out)
                switch(baud)
        if now >= next_tick:
            next_tick = now + args.interval
            if args.mode == "smart":
                out, baud = dev.tick(now)
                send(out)
                switch(baud)
            elif args.mode == "raw-nmea":
                send([nmea_sentence(rng)])
            else:
                ser.write(modbus_frame(rng))


if __name__ == "__main__":
    main()
