"""Prompt/target contract shared with the firmware (edge/action_schema.h,
edge/inference_task.h). Change both sides together."""

from __future__ import annotations

import json
from typing import Any

from .model_spec import (ACTION_MAX_TOKENS, CHAT_HISTORY_MAX_BYTES, CHAT_LISA_TAG, CHAT_MAX_CHARS, CHAT_USER_TAG,
                         EVENT_MAX_CHARS,
                         PROMPT_MAX_TOKENS, SPEC)
from .tokenizer import BpeTokenizer

NOP = "NOP"
KEY_MAX = 39      # DeviceProfile.key[40]
TYPE_MAX = 23     # DeviceProfile.type[24]
PROMPT_MAX_BYTES = 199   # InferRequest.prompt[200]


def build_prompt(device_key: str, device_type: str, event: str | bytes) -> bytes:
    """"<d>{key}:{type}<e>{event[:120], CR/LF -> space}<a>" as bytes."""
    ev = event.encode("utf-8") if isinstance(event, str) else bytes(event)
    ev = ev[:EVENT_MAX_CHARS].replace(b"\r", b" ").replace(b"\n", b" ")
    key = device_key.encode("utf-8")[:KEY_MAX]
    typ = (device_type or "").encode("utf-8")[:TYPE_MAX]
    return (b"<d>" + key + b":" + typ + b"<e>" + ev + b"<a>")[:PROMPT_MAX_BYTES]


def _chat_text(text: str | bytes) -> bytes:
    t = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    return t.replace(b"\r", b" ").replace(b"\n", b" ").strip(b" \t")[:CHAT_MAX_CHARS]


# Follow-ups: the only messages that get her previous reply as context. A 2.75M model
# given context on every message started copying her last reply back; gating it by
# message keeps ordinary questions single-turn. Same list in edge/action_schema.h.
FOLLOWUP_REPEAT = ("say that again", "repeat that", "what did you say", "can you repeat that", "pardon",
                   "come again", "say it again", "repeat please")
FOLLOWUP_MORE = ("tell me more", "why", "what do you mean", "explain", "go on", "how so",
                 "can you explain that", "tell me more about it")
FOLLOWUPS = FOLLOWUP_REPEAT + FOLLOWUP_MORE


def normalize_message(text: str | bytes) -> str:
    """Lowercase, trimmed, no end punctuation, no \"lisa, \" / \", lisa\", single spaces
    (edge_normalize_message)."""
    t = text.decode("utf-8", "replace") if isinstance(text, bytes) else text
    t = t.strip().lower().rstrip("?!. ")
    t = t.removeprefix("lisa, ").removeprefix("lisa ").removesuffix(", lisa").removesuffix(" lisa")
    return " ".join(t.split())


def is_followup(text: str | bytes) -> bool:
    return normalize_message(text) in FOLLOWUPS


def build_chat_prompt(text: str | bytes, prev_reply: str | bytes | None = None) -> bytes:
    """"<u>{text}<l>", or with memory "<l>{prev_reply}<u>{text}<l>" when the message is a
    follow-up and that fits CHAT_HISTORY_MAX_BYTES. Each part: CR/LF -> space, trimmed, cut to
    CHAT_MAX_CHARS bytes. Must match edge_build_chat_prompt() byte for byte."""
    t = _chat_text(text)
    if prev_reply and is_followup(t):
        p = _chat_text(prev_reply)
        with_history = CHAT_LISA_TAG + p + CHAT_USER_TAG + t + CHAT_LISA_TAG
        if p and len(with_history) <= CHAT_HISTORY_MAX_BYTES:
            return with_history
    return CHAT_USER_TAG + t + CHAT_LISA_TAG


def is_chat_prompt(prompt: bytes) -> bool:
    return prompt.startswith(CHAT_USER_TAG) or prompt.startswith(CHAT_LISA_TAG)


def split_chat_prompt(prompt: str) -> tuple[str | None, str]:
    """(previous reply or None, message) from a chat prompt."""
    body = prompt[:-3] if prompt.endswith("<l>") else prompt
    if body.startswith("<l>") and "<u>" in body:
        prev, msg = body[3:].split("<u>", 1)
        return prev, msg
    return None, body.removeprefix("<u>")


def canonical_action(cmd: str, args: dict[str, Any] | None) -> str:
    """The exact target text the model is trained to emit for an action."""
    return json.dumps({"cmd": cmd, "args": dict(sorted((args or {}).items()))}, separators=(",", ":"))


def canonicalize_output(text: str | None) -> str | None:
    """Normalizes a model output / label to the training target form, or None if
    it is neither NOP nor a JSON action."""
    if text is None:
        return None
    t = text.strip()
    if t.startswith(NOP):
        return NOP
    try:
        obj = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("cmd"), str):
        return None
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        return None
    return canonical_action(obj["cmd"], args)


def prompt_tokens(tok: BpeTokenizer, prompt: bytes) -> list[int]:
    """[BOS] + prompt tokens, cut to the device's budget (inference_task.h)."""
    body = tok.encode(prompt, max_tokens=200)[: PROMPT_MAX_TOKENS - 1]
    return [SPEC.bos_id] + body


def target_fits(tok: BpeTokenizer, target: str) -> bool:
    """The device stops after ACTION_MAX_TOKENS; a longer target can never be produced whole."""
    return len(tok.encode(target.encode("utf-8"))) <= ACTION_MAX_TOKENS


def training_sequence(tok: BpeTokenizer, prompt: bytes, target: str) -> tuple[list[int], int]:
    """Returns (tokens, n_prompt): [BOS] prompt target [EOS], loss on target+EOS only.
    Callers should drop examples that fail target_fits() rather than truncate them:
    a truncated target teaches the model to emit broken JSON."""
    p = prompt_tokens(tok, prompt)
    t = tok.encode(target.encode("utf-8"))[:ACTION_MAX_TOKENS] + [SPEC.eos_id]
    seq = (p + t)[: SPEC.max_context]
    return seq, len(p)
