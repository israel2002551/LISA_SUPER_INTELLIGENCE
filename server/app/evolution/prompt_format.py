"""Prompt/target contract shared with the firmware (edge/action_schema.h,
edge/inference_task.h). Change both sides together."""

from __future__ import annotations

import json
from typing import Any

from .model_spec import (ACTION_MAX_TOKENS, CHAT_LISA_TAG, CHAT_MAX_CHARS, CHAT_USER_TAG, EVENT_MAX_CHARS,
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


def build_chat_prompt(text: str | bytes) -> bytes:
    """"<u>{text}<l>": CR/LF -> space, trimmed, cut to CHAT_MAX_CHARS bytes (edge_build_chat_prompt)."""
    t = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    t = t.replace(b"\r", b" ").replace(b"\n", b" ").strip(b" \t")[:CHAT_MAX_CHARS]
    return CHAT_USER_TAG + t + CHAT_LISA_TAG


def is_chat_prompt(prompt: bytes) -> bool:
    return prompt.startswith(CHAT_USER_TAG)


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
