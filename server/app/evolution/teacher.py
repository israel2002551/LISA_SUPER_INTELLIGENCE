"""Teachers label what the edge model SHOULD have done for an event.

RuleTeacher   deterministic, free. Uses rules the peripheral declared in its
              manifest ("rules") plus rules an operator stored in
              uart_devices.learned_rules. Unidentified devices always get NOP.
ClaudeTeacher RuleTeacher first; when no rule decides, asks Claude (structured
              JSON output) and keeps the answer only if it passes the device's
              manifest validation. Capped per evolution run. Also classifies
              unknown UART streams for /api/v1/uart/classify.

Rule grammar:
  {"when": {"ev": "temp", "field": "v", "gt": 30}, "do": {"cmd": "set_fan", "args": {"level": 2}}}
  when: "ev" must equal the event's "ev"; "field" (default "v") is compared with
        any of gt / ge / lt / le / eq. "do": an action, or "NOP".
  Rules are checked in order; the first match wins. If rules exist and none
  match, the label is NOP.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from .action_schema import validate_action
from .prompt_format import NOP

log = logging.getLogger(__name__)


@dataclass
class DeviceContext:
    key: str
    type: str = "device"
    identified: bool = False
    data_format: str = "unknown"
    manifest: dict[str, Any] = field(default_factory=dict)
    rules: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "DeviceContext":
        manifest = row.get("capability_manifest") or {}
        rules = list(row.get("learned_rules") or []) + list(manifest.get("rules") or [])
        return cls(key=row["device_key"], type=row.get("device_type") or manifest.get("type") or "device",
                   identified=bool(row.get("identified")), data_format=row.get("data_format") or "unknown",
                   manifest=manifest, rules=rules)


@dataclass(frozen=True)
class Label:
    target: str          # canonical action JSON or "NOP"
    teacher: str
    confidence: float


class Teacher(Protocol):
    name: str

    def label(self, dev: DeviceContext, event: str) -> Label | None: ...


# ---------------------------------------------------------------------------
# Rule teacher
# ---------------------------------------------------------------------------
_OPS = {"gt": lambda a, b: a > b, "ge": lambda a, b: a >= b, "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b, "eq": lambda a, b: a == b}


def rule_matches(when: dict[str, Any], event: dict[str, Any]) -> bool:
    if "ev" in when and event.get("ev") != when["ev"]:
        return False
    value = event.get(when.get("field", "v"))
    for op, fn in _OPS.items():
        if op in when:
            if value is None or isinstance(value, bool):
                return False
            try:
                if not fn(value, when[op]):
                    return False
            except TypeError:
                return False
    return True


class RuleTeacher:
    name = "rule"

    def label(self, dev: DeviceContext, event: str) -> Label | None:
        if not dev.identified:
            return Label(NOP, self.name, 1.0)   # no command channel: never act
        if not dev.rules:
            return None
        try:
            ev = json.loads(event)
        except (json.JSONDecodeError, ValueError):
            return None
        if not isinstance(ev, dict):
            return None
        for rule in dev.rules:
            if not isinstance(rule, dict) or not rule_matches(rule.get("when") or {}, ev):
                continue
            do = rule.get("do")
            if do in (None, NOP, "nop"):
                return Label(NOP, self.name, 1.0)
            verdict, target, why = validate_action(dev.manifest, json.dumps(do))
            if verdict != "ok" or target is None:
                log.warning("Rule for %s produces an invalid action (%s): %s", dev.key, why, do)
                return None
            return Label(target, self.name, 1.0)
        return Label(NOP, self.name, 0.9)


# ---------------------------------------------------------------------------
# Claude teacher
# ---------------------------------------------------------------------------
class _ArgValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    value_json: str


class TeacherDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["action", "nop"]
    cmd: str
    args: list[_ArgValue]
    confidence: float
    rationale: str


class StreamClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["json_lines", "nmea", "ascii_lines", "csv", "modbus_rtu", "binary", "unknown"]
    device_type: str
    intent: str
    fields: list[str]
    framing: str
    confidence: float


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["action", "nop"]},
        "cmd": {"type": "string", "description": "Command name from the manifest; empty string for nop."},
        "args": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value_json": {"type": "string", "description": "The value as JSON, e.g. 2, 21.5, true, \"auto\"."},
                },
                "required": ["name", "value_json"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number", "description": "0 to 1."},
        "rationale": {"type": "string", "description": "One sentence."},
    },
    "required": ["decision", "cmd", "args", "confidence", "rationale"],
    "additionalProperties": False,
}

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "format": {"type": "string", "enum": ["json_lines", "nmea", "ascii_lines", "csv", "modbus_rtu", "binary", "unknown"]},
        "device_type": {"type": "string", "description": "Short snake_case device category, e.g. gps, thermostat."},
        "intent": {"type": "string", "description": "What this device most likely needs from the host, in a few words."},
        "fields": {"type": "array", "items": {"type": "string"}},
        "framing": {"type": "string", "description": "How messages are delimited."},
        "confidence": {"type": "number", "description": "0 to 1."},
    },
    "required": ["format", "device_type", "intent", "fields", "framing", "confidence"],
    "additionalProperties": False,
}

LABEL_SYSTEM = """You label training data for LISA, a tiny on-device model that controls hardware peripherals over UART.
Given the peripheral's capability manifest and one event it sent, decide what the controller should do next.
Rules:
- Only use commands listed in the manifest's "accepts", with every declared argument, each within its spec
  (int:MIN..MAX, float:MIN..MAX, bool, str:MAXLEN, enum:a|b).
- Choose "nop" when no action is needed, when the event is informational, or when you are unsure.
  Unneeded actions are worse than none: they move real hardware.
- confidence reflects how sure you are this is what the device's operator would want."""

CLASSIFY_SYSTEM = """You identify unknown devices from the first bytes they send on a UART line.
Given a hex dump, the baud rate and a heuristic format guess, describe the wire format, the device category,
what it most likely needs from a host controller, and the data fields you can see. Use format "unknown" and a low
confidence when the bytes don't support a conclusion."""

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"
REFUSAL_FALLBACKS = [{"model": "claude-opus-4-8"}]


class ClaudeTeacher:
    def __init__(self, client: Any, model: str = "claude-opus-5-5", effort: str = "medium", max_calls: int = 50,
                 min_confidence: float = 0.6) -> None:
        self.rules = RuleTeacher()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_calls = max_calls
        self.min_confidence = min_confidence
        self.calls = 0
        self.name = model

    @classmethod
    def from_env(cls, model: str, effort: str, max_calls: int) -> "ClaudeTeacher":
        import anthropic   # credentials: ANTHROPIC_API_KEY, or an `ant auth login` profile

        return cls(anthropic.Anthropic(), model=model, effort=effort, max_calls=max_calls)

    def reset_budget(self) -> None:
        self.calls = 0

    def _ask(self, system: str, user: str, schema: dict[str, Any]) -> str | None:
        """One structured-output request. Returns the JSON text, or None on any failure."""
        import anthropic

        if self.calls >= self.max_calls:
            return None
        self.calls += 1
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=[REFUSAL_FALLBACK_BETA],
                fallbacks=REFUSAL_FALLBACKS,
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.RateLimitError:
            log.warning("Claude teacher rate-limited; skipping this label")
            return None
        except anthropic.APIStatusError as e:
            log.warning("Claude teacher API error %s: %s", e.status_code, e.message)
            return None
        except anthropic.APIConnectionError:
            log.warning("Claude teacher could not reach the API")
            return None
        if response.stop_reason == "refusal":
            return None
        return next((b.text for b in response.content if b.type == "text"), None)

    def label(self, dev: DeviceContext, event: str) -> Label | None:
        ruled = self.rules.label(dev, event)
        if ruled is not None:
            return ruled
        user = (f"Manifest:\n{json.dumps(dev.manifest, sort_keys=True)}\n\n"
                f"Device: {dev.key} ({dev.type})\nEvent:\n{event}")
        text = self._ask(LABEL_SYSTEM, user, DECISION_SCHEMA)
        if text is None:
            return None
        try:
            d = TeacherDecision.model_validate_json(text)
        except ValidationError:
            return None
        if d.confidence < self.min_confidence:
            return None
        if d.decision == "nop":
            return Label(NOP, self.name, d.confidence)
        try:
            args = {a.name: json.loads(a.value_json) for a in d.args}
        except (json.JSONDecodeError, ValueError):
            return None
        verdict, target, _why = validate_action(dev.manifest, json.dumps({"cmd": d.cmd, "args": args}))
        return Label(target, self.name, d.confidence) if verdict == "ok" and target else None

    def classify_stream(self, sample: bytes, baud: int, format_guess: str) -> StreamClassification | None:
        user = f"Baud: {baud}\nHeuristic format guess: {format_guess}\nFirst {len(sample)} bytes (hex):\n{sample.hex()}"
        text = self._ask(CLASSIFY_SYSTEM, user, CLASSIFY_SCHEMA)
        if text is None:
            return None
        try:
            return StreamClassification.model_validate_json(text)
        except ValidationError:
            return None
