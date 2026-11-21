"""Teachers label what the edge model SHOULD have done for an event.

RuleTeacher   deterministic, free. Uses rules the peripheral declared in its
              manifest ("rules") plus rules an operator stored in
              uart_devices.learned_rules. Unidentified devices always get NOP.
LLMTeacher    (ClaudeTeacher: Claude; OpenAICompatibleTeacher: Groq or any
              OpenAI-compatible API) RuleTeacher first; when no rule decides, asks the model (structured
              JSON output) and keeps the answer only if it passes the device's
              manifest validation. Capped per evolution run. Also classifies
              unknown UART streams for /api/v1/uart/classify, reviews LISA's
              real chat replies (good, or the reply she should have given),
              and teaches her new conversation topics (lessons).

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

# ---------------------------------------------------------------------------
# Chat teaching: review LISA's real replies, and teach her new topics
# ---------------------------------------------------------------------------
LISA_PERSONA = """LISA is a tiny AI: a 2.75-million-parameter language model running on an ESP32-S3 microcontroller,
with her weights on a W25Q128 flash chip. She controls a smart thermostat: above 30 degrees she sets the fan to
level 2, below 15 degrees to level 0, otherwise she does nothing. She learns because her evolution server retrains
her on what happens and sends signed updates over Wi-Fi; she tests each update and rolls back if it fails. She chats
over a USB serial console and remembers only her previous answer. She can't see, hear, move, browse the web or read
a clock, and she only understands simple English. She is friendly, honest about her limits, and brief."""

REPLY_RULES = """A reply LISA should give is: one line, at most 120 characters, plain ASCII, in the first person as
LISA, in simple words. For simple, stable facts give the short correct answer (e.g. "The capital of France is
Paris."). For things she can't know or do (news, prices, the current time, the web, political or religious
opinions), say so kindly."""

CHAT_REVIEW_SYSTEM = f"""You review LISA's replies in real conversations; your reviews become her training data.
{LISA_PERSONA}

Each numbered turn has her previous reply (if any), the user's message, and LISA's reply.
- verdict "good": a sensible, truthful, friendly answer to the message in LISA's voice. It doesn't have to be the
  reply you would write.
- verdict "bad": off-topic, wrong, garbled, repetitive or unhelpful.
- better_reply: the reply LISA should give. For a "good" verdict, repeat her reply exactly.
- confidence: 0 to 1.
{REPLY_RULES}"""

LESSONS_SYSTEM = f"""You teach LISA new conversation topics; your lessons become her training data.
{LISA_PERSONA}

Give each topic a short name, 5 to 8 different ways a person might say it (lowercase, no end punctuation, under
60 characters each), and the one reply LISA should give. Choose useful, everyday topics a friendly home assistant
should handle: simple stable facts, kindness, everyday help, fun facts. Every phrasing must clearly belong to its
own topic only. Skip time-sensitive facts, opinions, and anything she can't do.
Only teach general knowledge. Never teach anything about LISA herself, her devices, her commands or what she can
do: her own lessons cover that, and a wrong claim about herself is worse than none.
{REPLY_RULES}"""

CHAT_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["good", "bad"]},
                    "better_reply": {"type": "string"},
                    "confidence": {"type": "number", "description": "0 to 1."},
                },
                "required": ["index", "verdict", "better_reply", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reviews"],
    "additionalProperties": False,
}

LESSONS_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "phrasings": {"type": "array", "items": {"type": "string"}},
                    "reply": {"type": "string"},
                },
                "required": ["topic", "phrasings", "reply"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lessons"],
    "additionalProperties": False,
}


class ChatReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    verdict: Literal["good", "bad"]
    better_reply: str
    confidence: float


class _ChatReviews(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviews: list[ChatReview]


class Lesson(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str
    phrasings: list[str]
    reply: str


class _Lessons(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lessons: list[Lesson]


@dataclass(frozen=True)
class ChatTurn:
    prev_reply: str | None
    message: str
    reply: str


def clean_reply(text: str, max_chars: int = 120) -> str | None:
    """A reply fit to train on: one line, printable ASCII, 1..max_chars characters."""
    t = " ".join((text or "").split())
    if not t or len(t) > max_chars or not all(32 <= ord(c) < 127 for c in t):
        return None
    return t


REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"
REFUSAL_FALLBACKS = [{"model": "claude-opus-4-8"}]


class LLMTeacher:
    """What every API teacher does, given one structured-JSON call (_ask):
    device labels (rules first), chat reviews, lessons, and UART stream classification.
    Each call counts against max_calls per evolution run."""

    def __init__(self, model: str, max_calls: int = 50, min_confidence: float = 0.6) -> None:
        self.rules = RuleTeacher()
        self.model = model
        self.max_calls = max_calls
        self.min_confidence = min_confidence
        self.calls = 0
        self.name = model

    def reset_budget(self) -> None:
        self.calls = 0

    def _take_call(self) -> bool:
        if self.calls >= self.max_calls:
            return False
        self.calls += 1
        return True

    def _ask(self, system: str, user: str, schema: dict[str, Any]) -> str | None:
        """One structured-output request. Returns the JSON text, or None on any failure."""
        raise NotImplementedError

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

    def review_chats(self, turns: list[ChatTurn], batch: int = 20) -> list[ChatReview | None]:
        """One review per turn (None where the model gave none, or an unusable one)."""
        out: list[ChatReview | None] = [None] * len(turns)
        for start in range(0, len(turns), batch):
            chunk = turns[start:start + batch]
            user = "\n\n".join(
                f"Turn {i}:\nHer previous reply: {t.prev_reply or '(none)'}\nUser: {t.message}\nLISA: {t.reply or '(no reply)'}"
                for i, t in enumerate(chunk))
            text = self._ask(CHAT_REVIEW_SYSTEM, user, CHAT_REVIEW_SCHEMA)
            if text is None:
                continue
            try:
                reviews = _ChatReviews.model_validate_json(text).reviews
            except ValidationError:
                continue
            for r in reviews:
                better = clean_reply(r.better_reply)
                if 0 <= r.index < len(chunk) and better and r.confidence >= self.min_confidence:
                    out[start + r.index] = r.model_copy(update={"better_reply": better})
        return out

    def lessons(self, n: int, known_topics: list[str]) -> list[Lesson]:
        """Up to n new topics to teach, each with phrasings and LISA's reply."""
        if n <= 0:
            return []
        user = f"Teach LISA {n} new topics. She already knows: {', '.join(known_topics)}."
        text = self._ask(LESSONS_SYSTEM, user, LESSONS_SCHEMA)
        if text is None:
            return []
        try:
            lessons = _Lessons.model_validate_json(text).lessons
        except ValidationError:
            return []
        good = []
        for lesson in lessons[:n]:
            reply = clean_reply(lesson.reply)
            phrasings = [p for p in (clean_reply(p.lower().rstrip("?!. "), 60) for p in lesson.phrasings) if p]
            if reply and len(phrasings) >= 3 and lesson.topic.strip():
                good.append(Lesson(topic=lesson.topic.strip()[:40], phrasings=phrasings[:8], reply=reply))
        return good

    def classify_stream(self, sample: bytes, baud: int, format_guess: str) -> StreamClassification | None:
        user = f"Baud: {baud}\nHeuristic format guess: {format_guess}\nFirst {len(sample)} bytes (hex):\n{sample.hex()}"
        text = self._ask(CLASSIFY_SYSTEM, user, CLASSIFY_SCHEMA)
        if text is None:
            return None
        try:
            return StreamClassification.model_validate_json(text)
        except ValidationError:
            return None


class ClaudeTeacher(LLMTeacher):
    """Claude through the Anthropic API (ANTHROPIC_API_KEY)."""

    def __init__(self, client: Any, model: str = "claude-opus-5-5", effort: str = "medium", max_calls: int = 50,
                 min_confidence: float = 0.6) -> None:
        super().__init__(model, max_calls, min_confidence)
        self.client = client
        self.effort = effort

    @classmethod
    def from_env(cls, model: str, effort: str, max_calls: int) -> "ClaudeTeacher":
        import anthropic   # credentials: ANTHROPIC_API_KEY, or an `ant auth login` profile

        return cls(anthropic.Anthropic(), model=model, effort=effort, max_calls=max_calls)

    def _ask(self, system: str, user: str, schema: dict[str, Any]) -> str | None:
        import anthropic

        if not self._take_call():
            return None
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


GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"   # production model with strict JSON-schema outputs on Groq


class OpenAICompatibleTeacher(LLMTeacher):
    """Any OpenAI-compatible chat-completions API with strict JSON-schema outputs; Groq by default
    (GROQ_API_KEY, free plan with rate limits). On HTTP 429 it waits out `retry-after` (up to
    max_wait_s) and retries, so a burst of calls in one evolution run doesn't lose reviews."""

    def __init__(self, api_key: str, model: str = GROQ_DEFAULT_MODEL, base_url: str = GROQ_BASE_URL,
                 max_calls: int = 50, min_confidence: float = 0.6, http: Any = None, max_wait_s: float = 90.0,
                 sleep: Any = None) -> None:
        import time

        import httpx

        super().__init__(model, max_calls, min_confidence)
        self.name = f"groq/{model}" if base_url == GROQ_BASE_URL else model
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=120)
        self.max_wait_s = max_wait_s
        self.sleep = sleep or time.sleep

    def _ask(self, system: str, user: str, schema: dict[str, Any]) -> str | None:
        import httpx

        if not self._take_call():
            return None
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "lisa_teacher", "strict": True, "schema": schema}},
        }
        for _attempt in range(4):
            try:
                r = self.http.post(f"{self.base_url}/chat/completions", json=body)
            except httpx.TransportError as e:
                log.warning("Teacher %s could not reach the API: %s", self.name, e)
                return None
            if r.status_code == 429:
                try:
                    wait = float(r.headers.get("retry-after", "10"))
                except ValueError:
                    wait = 10.0
                if wait > self.max_wait_s:
                    log.warning("Teacher %s rate-limited for %.0f s; skipping", self.name, wait)
                    return None
                log.info("Teacher %s rate-limited; waiting %.0f s", self.name, wait)
                self.sleep(wait + 0.5)
                continue
            if r.status_code != 200:
                log.warning("Teacher %s API error %s: %s", self.name, r.status_code, r.text[:300])
                return None
            try:
                return r.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError, ValueError):
                log.warning("Teacher %s returned an unexpected response", self.name)
                return None
        return None
