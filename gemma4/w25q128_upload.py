"""
w25q128_upload.py - writes a model image to the external W25Q128 through the ESP32-S3
====================================================================================
The W25Q128 is wired to the ESP32-S3's SPI2 pins, so esptool can't program it.
Instead the running firmware (env gemma4_w25q128_cluster) takes the image over
the serial port, writes it to the chip, reads it all back to check the CRC, and
then loads it into PSRAM.

Usage (close any serial monitor on the port first):
  python gemma4/w25q128_upload.py COM3 cluster_binaries/gemma4_w25q128_int16.bin

Expect roughly 1-2 minutes for the default ~5.3 MB image at 921600 baud
(erase, then 4 KB chunks, each acknowledged by the board).
"""

import argparse
import re
import struct
import sys
import time
import zlib

try:
    import serial
except ImportError:
    sys.exit("pyserial is required: pip install pyserial "
             "(or run this with PlatformIO's Python, which already has it)")

MODEL_MAGIC = 0x31573447   # "G4W1", W25_MODEL_MAGIC in gemma4_w25q128_config.h
HEADER_FMT = "<IHH8I6I"
EVENT = re.compile(r"@W25 (\w+)\s*(.*)")


class Device:
    """Reads the firmware's output line by line and picks out '@W25 ...' events."""

    def __init__(self, ser):
        self.ser = ser
        self.buf = b""

    def event(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            while b"\n" in self.buf:
                raw, self.buf = self.buf.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").strip()
                m = EVENT.search(line)
                if m:
                    return m.group(1), m.group(2).strip()
                if line:
                    print(f"  | {line}")
            if time.monotonic() > deadline:
                raise TimeoutError("no response from the board")
            self.buf += self.ser.read(self.ser.in_waiting or 1)

    def drain_until_quiet(self, quiet=0.5, limit=30.0):
        """Opening the port can reset some boards; wait for boot output to stop."""
        start = last = time.monotonic()
        while time.monotonic() - start < limit:
            data = self.ser.read(self.ser.in_waiting or 1)
            if data:
                last = time.monotonic()
            elif time.monotonic() - last >= quiet:
                return
        self.buf = b""


def describe_image(data):
    if len(data) < struct.calcsize(HEADER_FMT):
        sys.exit("error: file is too small to be a model image")
    (magic, version, header_bytes, hidden, inter, heads, layers, vocab,
     payload_bytes, payload_crc, flags, *_) = struct.unpack_from(HEADER_FMT, data)
    if magic != MODEL_MAGIC:
        sys.exit("error: not a model image (bad magic). Build one with gemma4/gemma4_slice_int16.py")
    if header_bytes + payload_bytes != len(data):
        sys.exit(f"error: header says {header_bytes + payload_bytes} bytes, file has {len(data)}")
    print(f"Image v{version}: H={hidden} FFN={inter} heads={heads} layers={layers} vocab={vocab}, "
          f"{len(data):,} bytes")


def fail(msg):
    print(f"\nFAILED: {msg}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Write a model image to the external W25Q128 via the ESP32-S3")
    parser.add_argument("port", help="serial port, e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("image", help="image built by gemma4_slice_int16.py")
    parser.add_argument("--baud", type=int, default=921600, help="must match SERIAL_BAUD in the firmware")
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        data = f.read()
    describe_image(data)
    crc = zlib.crc32(data)

    ser = serial.Serial()
    ser.port = args.port
    ser.baudrate = args.baud
    ser.timeout = 0.2
    ser.dtr = False   # keep EN/IO0 released so opening the port doesn't reset the board
    ser.rts = False
    ser.open()
    dev = Device(ser)
    dev.drain_until_quiet()

    ser.write(f"/w25write {len(data)} {crc:08X}\n".encode())

    # Erase, then READY
    while True:
        kind, rest = dev.event(timeout=60)
        if kind == "ERR":
            fail(rest)
        if kind == "ERASE":
            print(f"\rErasing blocks {rest} ", end="", flush=True)
        elif kind == "READY":
            chunk = int(rest)
            print()
            break

    # Data, one chunk per ACK
    sent = 0
    start = time.monotonic()
    while sent < len(data):
        block = data[sent:sent + chunk]
        ser.write(block)
        kind, rest = dev.event(timeout=15)
        if kind == "ERR":
            fail(rest)
        if kind != "ACK" or int(rest) != sent + len(block):
            fail(f"unexpected reply '{kind} {rest}' after {sent + len(block)} bytes")
        sent += len(block)
        rate = sent / max(time.monotonic() - start, 1e-6) / 1024
        print(f"\rWriting {sent * 100 // len(data):3d}%  ({sent:,}/{len(data):,} bytes, {rate:.0f} KB/s)",
              end="", flush=True)
    print()

    # Read-back check, then the firmware loads the model
    kind, rest = dev.event(timeout=120)
    if kind != "DONE":
        fail(rest)
    print(f"Verified on chip (CRC {rest}). Loading into PSRAM...")
    kind, rest = dev.event(timeout=120)
    if kind != "LOADED":
        fail(rest)
    print("Done: the model is on the W25Q128 and loaded.")
    ser.close()


if __name__ == "__main__":
    main()
