"""Synthetic data & distillation: targeted examples the logs don't cover yet.

For every identified device with rules, sweep event values around each rule's
threshold (where a small model is most likely to be wrong), then fill the
device's quota with values spread evenly over the whole range around its rules,
and label them all with the rule teacher. The boundary sweep alone is not enough:
on hardware, a model trained on ~40 near-threshold examples learned only the
majority answer (NOP for every temperature). Coverage of the full range is what
teaches it that the value decides the action.

For events the device emits that no rule covers, and for unidentified/raw
devices, teach NOP: the model must not invent actions.

The random draws are seeded from the device, not the run, so re-synthesizing an
unchanged device yields the same rows (deduplicated on insert) instead of growing
the dataset on every evolution run.

Events use the convention the edge protocol documents: {"ev":<name>,"v":<value>}
in compact JSON (no spaces), which is what tools/fake_peripheral.py sends.
"""

from __future__ import annotations

import json
import random
from typing import Any

from .curator import example_row
from .prompt_format import NOP, build_prompt
from .teacher import DeviceContext, RuleTeacher

RAW_SAMPLES = {
    "nmea": ["$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47",
             "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"],
    "csv": ["21.4,48.2,1013.1", "21.5,48.0,1013.0"],
    "ascii_lines": ["READY", "ERR 4", "OK"],
    "json_lines": ['{"t":21.4}', '{"status":"idle"}'],
    "modbus_rtu": ["01030400640032ba1d"],
    "binary": ["a5 5a 03 10 ff"],
}


def _event(name: str, field: str, value: Any) -> str:
    return json.dumps({"ev": name, field: value}, separators=(",", ":"))


def _numeric_rules(dev: DeviceContext) -> list[tuple[str, str, float]]:
    out = []
    for rule in dev.rules:
        when = (rule or {}).get("when") or {}
        for op in ("gt", "ge", "lt", "le"):
            if op in when and isinstance(when[op], (int, float)) and "ev" in when:
                out.append((when["ev"], when.get("field", "v"), float(when[op])))
    return out


def _value_ranges(rules: list[tuple[str, str, float]]) -> dict[tuple[str, str], tuple[float, float]]:
    """Per (event, field): the span of its thresholds, padded on both sides so every
    region a rule separates is covered (e.g. lt 15 / gt 30 -> 0 .. 45)."""
    thresholds: dict[tuple[str, str], list[float]] = {}
    for ev, field, thr in rules:
        thresholds.setdefault((ev, field), []).append(thr)
    ranges = {}
    for key, thrs in thresholds.items():
        lo, hi = min(thrs), max(thrs)
        pad = max(10.0, abs(lo) * 0.5, abs(hi) * 0.5, hi - lo)
        ranges[key] = (lo - pad, hi + pad)
    return ranges


def synthesize(devices: list[DeviceContext], per_device: int, seed: int = 0) -> list[dict]:
    rules = RuleTeacher()
    rows: list[dict] = []
    for dev in devices:
        rng = random.Random(f"{seed}:{dev.key}:{json.dumps(dev.manifest, sort_keys=True)}")
        produced: dict[str, str] = {}   # event -> target, insertion-ordered, no duplicates

        def add(event: str) -> None:
            label = rules.label(dev, event)
            if label is not None and event not in produced:
                produced[event] = label.target

        if not dev.identified:
            for sample in RAW_SAMPLES.get(dev.data_format, RAW_SAMPLES["ascii_lines"]):
                produced.setdefault(sample, NOP)
        else:
            numeric = _numeric_rules(dev)
            for ev, field, thr in numeric:
                span = max(1.0, abs(thr) * 0.1)
                offsets = [-10, -5, -2, -1, -0.5, -0.1, 0.0, 0.1, 0.5, 1, 2, 5, 10]
                for v in [thr + o * span / 2 for o in offsets] + [thr + rng.uniform(-3, 3) * span for _ in range(6)]:
                    add(_event(ev, field, round(v, 1)))
            covered = {ev for ev, _, _ in numeric}
            for name in dev.manifest.get("emits") or []:
                if isinstance(name, str) and name not in covered:
                    for _ in range(3):
                        add(_event(name, "v", round(rng.uniform(-10, 100), 1)))
            ranges = list(_value_ranges(numeric).items())
            attempts = 0
            while ranges and len(produced) < per_device and attempts < per_device * 4:
                (ev, field), (lo, hi) = ranges[attempts % len(ranges)]
                add(_event(ev, field, round(rng.uniform(lo, hi), 1)))
                attempts += 1

        for event, target in list(produced.items())[:per_device]:
            prompt = build_prompt(dev.key, dev.type, event).decode("utf-8", "replace")
            rows.append(example_row(prompt, target, "synthetic_distill", "rule", 1.0,
                                    context={"device_key": dev.key, "event": event}))
    return rows
