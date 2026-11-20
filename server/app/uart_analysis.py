"""Heuristic analysis of unidentified UART streams (cloud side of
analyze_raw() in edge/uart_link.h, with field extraction on top)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class StreamAnalysis:
    format: str
    device_type: str = "unknown"
    intent: str = "unknown"
    confidence: float = 0.3
    fields: list[str] = field(default_factory=list)
    framing: str = "unknown"
    source: str = "heuristic"

    def as_dict(self) -> dict:
        return asdict(self)


def _modbus_crc(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _is_text(b: int) -> bool:
    return 0x20 <= b < 0x7F or b in (0x0A, 0x0D, 0x09)


def analyze_stream(sample: bytes) -> StreamAnalysis:
    if not sample:
        return StreamAnalysis("unknown", confidence=0.0)
    text_ratio = sum(_is_text(b) for b in sample) / len(sample)
    if text_ratio >= 0.9:
        lines = [l.strip() for l in sample.decode("ascii", "replace").splitlines() if l.strip()]
        if not lines:
            return StreamAnalysis("ascii_lines", confidence=0.2, framing="newline")
        json_objs = []
        for l in lines:
            if l.startswith("{") and l.endswith("}"):
                try:
                    json_objs.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
        if len(json_objs) * 2 >= len(lines):
            keys = sorted({k for o in json_objs if isinstance(o, dict) for k in o})
            return StreamAnalysis("json_lines", "json_sensor", "telemetry ingestion", 0.8, keys, "newline")
        nmea = [l for l in lines if l.startswith("$") and "*" in l]
        if len(nmea) * 2 >= len(lines):
            sentences = sorted({l[1:6] for l in nmea})
            gps = any(s[2:] in ("GGA", "RMC", "GLL", "VTG") for s in sentences)
            return StreamAnalysis("nmea", "gps" if gps else "nmea_device", "position reporting" if gps else "telemetry",
                                  0.9 if gps else 0.7, sentences, "newline, $...*CS")
        csv = [l for l in lines if l.count(",") >= 2]
        if len(csv) * 2 >= len(lines):
            cols = max(l.count(",") + 1 for l in csv)
            return StreamAnalysis("csv", "csv_sensor", "periodic measurements", 0.6,
                                  [f"col{i}" for i in range(cols)], "newline, comma-separated")
        return StreamAnalysis("ascii_lines", "text_console", "status messages", 0.4, [], "newline")
    for n in range(4, min(len(sample), 256) + 1):
        crc = _modbus_crc(sample[: n - 2])
        if sample[n - 2] == crc & 0xFF and sample[n - 1] == crc >> 8:
            return StreamAnalysis("modbus_rtu", "modbus_slave", "register polling", 0.75,
                                  [f"slave={sample[0]}", f"function={sample[1]}"], f"RTU frame, {n} bytes, CRC-16")
    return StreamAnalysis("binary", confidence=0.2, framing="unknown binary")
