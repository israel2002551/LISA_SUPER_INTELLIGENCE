"""Telemetry ingestion: device status + ring-log events -> Supabase rows.

Idempotent: interaction_logs is unique on (node_id, device_seq), so a device
that re-uploads after a lost ack creates no duplicates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .evolution.curator import initial_reward
from .evolution.pipeline import EvolutionContext
from .schemas import TelemetryAck, TelemetryBatch
from .uart_analysis import analyze_stream

EVENT_KINDS = {"interaction", "device", "handshake", "observation", "feedback", "ota", "boot", "error"}
KIND_BY_RECORD_TYPE = {1: "interaction", 2: "device", 3: "error", 4: "ota", 5: "boot", 6: "feedback",
                       7: "observation", 8: "handshake"}
STATUSES = {"ok", "nop", "invalid_output", "rejected", "timeout", "error"}
BUSES = {"uart", "uart_fast", "espnow", "tcp", "spi"}


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _str(v: Any, limit: int = 4000) -> str | None:
    return v[:limit] if isinstance(v, str) else None


def device_fields(d: dict[str, Any]) -> dict[str, Any]:
    """uart_devices columns from a telemetry device entry or a 'device' event."""
    fields: dict[str, Any] = {
        "identified": bool(d.get("identified")),
        "port": _int(d.get("port")) or 1,
        "baud": _int(d.get("baud")),
        "bus": d.get("link") if d.get("link") in BUSES else "uart",
        "data_format": _str(d.get("format"), 32) or "unknown",
    }
    manifest = d.get("manifest")
    if manifest is None and isinstance(d.get("manifest_text"), str):
        try:
            manifest = json.loads(d["manifest_text"])
        except json.JSONDecodeError:
            manifest = None
    if isinstance(manifest, dict):
        fields["capability_manifest"] = manifest
        needs = manifest.get("needs")
        if isinstance(needs, list) and needs:
            fields["inferred_intent"] = ", ".join(str(n) for n in needs)[:200]
    if _str(d.get("type"), 64) and d.get("type") != "unknown":
        fields["device_type"] = d["type"][:64]
    sample = d.get("sample_hex")
    if isinstance(sample, str) and sample and not fields["identified"]:
        try:
            analysis = analyze_stream(bytes.fromhex(sample))
        except ValueError:
            analysis = None
        if analysis:
            fields["protocol_schema"] = analysis.as_dict()
            fields.setdefault("device_type", analysis.device_type)
            fields["inferred_intent"] = analysis.intent
    return fields


def ingest_telemetry(ctx: EvolutionContext, node: dict[str, Any], batch: TelemetryBatch) -> TelemetryAck:
    repo = ctx.repo
    node_id = node["id"]
    st = batch.status
    parts = st.get("partitions") if isinstance(st.get("partitions"), dict) else {}
    active_slot = parts.get("active_slot") if parts.get("active_slot") in (0, 1) else None
    slots = parts.get("slots") if isinstance(parts.get("slots"), list) else []
    active_sha = slots[active_slot].get("sha256") if active_slot is not None and len(slots) > active_slot else None
    mv = _int(st.get("model_version"))
    w25 = st.get("w25q") if isinstance(st.get("w25q"), dict) else {}
    repo.update_node(node_id, {
        "fw_version": batch.fw_version, "status": st, "partition_state": parts, "active_slot": active_slot,
        "active_model_version": mv if mv and repo.get_model(mv) else None,
        "active_model_sha256": active_sha or None, "w25q_jedec_id": _str(w25.get("jedec_id"), 16),
        "w25q_total_bytes": _int(w25.get("size")), "last_heartbeat_at": datetime.now(timezone.utc).isoformat(),
    })

    device_ids: dict[str, str] = {}

    def device_id(key: str | None, fields: dict[str, Any] | None = None) -> str | None:
        if not key:
            return None
        if fields is not None:
            device_ids[key] = repo.upsert_uart_device(node_id, key, fields)["id"]
        elif key not in device_ids:
            existing = repo.get_uart_device(node_id, key)
            device_ids[key] = existing["id"] if existing else repo.upsert_uart_device(node_id, key, {})["id"]
        return device_ids[key]

    for d in batch.devices:
        device_id(d.device_key, device_fields(d.model_dump()))

    rows: list[dict[str, Any]] = []
    feedback: list[tuple[int, float]] = []
    for ev in batch.events:
        data = ev.data
        kind = data.get("kind") if data.get("kind") in EVENT_KINDS else KIND_BY_RECORD_TYPE.get(ev.type, "error")
        dev_key = _str(data.get("dev"), 64)
        dev_id = device_id(dev_key, device_fields(data)) if kind == "device" else device_id(dev_key)
        row: dict[str, Any] = {"node_id": node_id, "device_seq": ev.seq, "event_kind": kind, "payload": data,
                               "uart_device_id": dev_id, "used_for_training": False}
        if kind == "interaction":
            status = data.get("status") if data.get("status") in STATUSES else "error"
            row.update(raw_input=_str(data.get("input")), prompt=_str(data.get("prompt")),
                       model_output=_str(data.get("output")),
                       action=data.get("action") if isinstance(data.get("action"), dict) else None,
                       model_version=_int(data.get("mv")), latency_ms=_int(data.get("lat")), status=status,
                       error_detail=_str(data.get("err"), 500), act_id=_int(data.get("act_id")),
                       reward=initial_reward(status))
        elif kind == "feedback" and _int(data.get("ref_act_id")):
            value = data.get("value")
            if value in ("good", "bad"):
                feedback.append((data["ref_act_id"], 1.0 if value == "good" else 0.0))
        rows.append(row)

    inserted = repo.insert_logs(rows)
    for act_id, reward in feedback:      # after insert: the interaction may be in this same batch
        repo.set_reward_by_act(node_id, act_id, reward)
    acked = max((e.seq for e in batch.events), default=0)
    return TelemetryAck(acked_seq=acked, inserted=inserted)
