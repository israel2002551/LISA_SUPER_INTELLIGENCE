"""Endpoints the ESP32-S3 calls (node API key auth)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..evolution.teacher import LLMTeacher
from ..ingest import ingest_telemetry
from ..schemas import ClassifyRequest, ClassifyResponse, OtaCheckResponse, TelemetryAck, TelemetryBatch
from ..security import Ctx, Node
from ..uart_analysis import analyze_stream

router = APIRouter(tags=["device"])


@router.post("/telemetry", response_model=TelemetryAck)
def telemetry(batch: TelemetryBatch, node: Node, ctx: Ctx) -> TelemetryAck:
    if batch.node_id != node["id"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "node_id doesn't match the authenticated node")
    return ingest_telemetry(ctx, node, batch)


@router.post("/uart/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest, node: Node, ctx: Ctx) -> ClassifyResponse:
    """Fast cloud assist for an unidentified UART stream: heuristics first, then
    Claude (TEACHER=claude) when the heuristics aren't confident."""
    if req.node_id != node["id"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "node_id doesn't match the authenticated node")
    sample = bytes.fromhex(req.sample_hex) if len(req.sample_hex) % 2 == 0 else b""
    result = analyze_stream(sample).as_dict()
    if result["confidence"] < 0.7 and isinstance(ctx.teacher, LLMTeacher):
        llm = ctx.teacher.classify_stream(sample, req.baud, req.format_guess)
        if llm is not None and llm.confidence > result["confidence"]:
            result = {**llm.model_dump(), "source": ctx.teacher.model}
    ctx.repo.upsert_uart_device(node["id"], req.device_key, {
        "baud": req.baud or None, "data_format": result["format"], "device_type": result["device_type"],
        "inferred_intent": result["intent"], "protocol_schema": result,
    })
    return ClassifyResponse(**result)


@router.get("/ota/check", response_model=OtaCheckResponse)
def ota_check(node_id: str, node: Node, ctx: Ctx, current_version: int = 0, current_sha256: str = "") -> OtaCheckResponse:
    """Is there a model the device should install? Prefers a block delta when
    the device runs exactly the delta's base image."""
    if node_id != node["id"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "node_id doesn't match the authenticated node")
    active = ctx.repo.get_active_model()
    if not active or current_sha256.lower() == active["sha256"]:
        return OtaCheckResponse(update=False, version=active["version"] if active else None)
    kind, url = "full", ctx.blobs.signed_url("models", active["storage_path"])
    if active.get("delta_from_version") and active.get("delta_storage_path"):
        base = ctx.repo.get_model(active["delta_from_version"])
        if base and base["sha256"] == current_sha256.lower():
            kind, url = "delta", ctx.blobs.signed_url("deltas", active["delta_storage_path"])
    return OtaCheckResponse(update=True, kind=kind, version=active["version"], size=active["size_bytes"],
                            sha256=active["sha256"], signature=active.get("signature"), url=url)
