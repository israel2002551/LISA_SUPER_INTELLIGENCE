"""Manifest parsing and action validation: Python twin of edge/action_schema.h.

Arg specs: "int" | "int:MIN..MAX" | "float" | "float:MIN..MAX" | "bool" |
           "str" | "str:MAXLEN" | "enum:a|b|c"
"""

from __future__ import annotations

import json
from typing import Any

from .prompt_format import NOP, canonical_action


def accepts_of(manifest: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    """{cmd: {arg_name: spec}} from an IDENTIFY manifest."""
    out: dict[str, dict[str, str]] = {}
    for c in (manifest or {}).get("accepts") or []:
        if isinstance(c, dict) and isinstance(c.get("cmd"), str) and c["cmd"]:
            args = c.get("args") or {}
            out[c["cmd"]] = {str(k): str(v) for k, v in args.items()} if isinstance(args, dict) else {}
    return out


def _range(param: str) -> tuple[float, float] | None:
    lo, sep, hi = param.partition("..")
    if not sep:
        return None
    try:
        return float(lo), float(hi)
    except ValueError:
        return None


def arg_matches(spec: str, value: Any) -> bool:
    kind, _, param = spec.partition(":")
    if kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        r = _range(param) if param else None
        return not param or (r is not None and r[0] <= value <= r[1])
    if kind == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        r = _range(param) if param else None
        return not param or (r is not None and r[0] <= value <= r[1])
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "str":
        return isinstance(value, str) and (not param or len(value.encode()) <= int(param))
    if kind == "enum":
        return isinstance(value, str) and value in param.split("|")
    return False


def validate_action(manifest: dict[str, Any] | None, text: str | None) -> tuple[str, str | None, str]:
    """Returns (verdict, canonical_target_or_None, why). verdict: ok | nop | invalid."""
    t = (text or "").lstrip(" ")
    if t.startswith(NOP):
        return "nop", NOP, ""
    try:
        obj = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return "invalid", None, "not JSON"
    if not isinstance(obj, dict):
        return "invalid", None, "not JSON"
    cmd = obj.get("cmd") if isinstance(obj.get("cmd"), str) else ""
    specs = accepts_of(manifest)
    if cmd not in specs:
        return "invalid", None, f"unknown cmd '{cmd}'"
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        return "invalid", None, "args not an object"
    for name, value in args.items():
        if name not in specs[cmd]:
            return "invalid", None, f"unknown arg '{name}'"
        if not arg_matches(specs[cmd][name], value):
            return "invalid", None, f"arg '{name}' violates {specs[cmd][name]}"
    for name in specs[cmd]:
        if name not in args:
            return "invalid", None, f"missing arg '{name}'"
    return "ok", canonical_action(cmd, args), ""
