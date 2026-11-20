"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

NODE_ID_PATTERN = r"^[A-Za-z0-9_-]{1,40}$"


class TelemetryEvent(BaseModel):
    seq: int = Field(ge=1)
    type: int
    data: dict[str, Any]


class TelemetryDevice(BaseModel):
    device_key: str = Field(min_length=1, max_length=64)
    port: int = 1
    identified: bool = False
    baud: int | None = None
    link: str = "uart"
    format: str = "unknown"
    type: str | None = None
    manifest: dict[str, Any] | None = None


class TelemetryBatch(BaseModel):
    node_id: str = Field(pattern=NODE_ID_PATTERN)
    fw_version: str | None = None
    status: dict[str, Any] = Field(default_factory=dict)
    devices: list[TelemetryDevice] = Field(default_factory=list)
    events: list[TelemetryEvent] = Field(default_factory=list, max_length=2000)


class TelemetryAck(BaseModel):
    acked_seq: int
    inserted: int
    rediscover: bool = False


class ClassifyRequest(BaseModel):
    node_id: str = Field(pattern=NODE_ID_PATTERN)
    device_key: str = Field(min_length=1, max_length=64)
    baud: int = 0
    format_guess: str = "unknown"
    sample_hex: str = Field(default="", max_length=1024, pattern=r"^[0-9a-fA-F]*$")


class ClassifyResponse(BaseModel):
    format: str
    device_type: str
    intent: str
    confidence: float
    fields: list[str]
    framing: str
    source: str


class OtaCheckResponse(BaseModel):
    update: bool
    rollback: bool = False
    kind: Literal["full", "delta"] | None = None
    version: int | None = None
    size: int | None = None
    sha256: str | None = None
    signature: str | None = None
    url: str | None = None


class TriggerRequest(BaseModel):
    force_publish: bool = False
    trainer: Literal["mock", "torch", "remote"] | None = None
    method: Literal["full", "lora"] | None = None
    steps: int | None = Field(default=None, ge=1, le=20000)


class TriggerResponse(BaseModel):
    run_id: str
    status: str
    executor: Literal["inline", "worker"]


class RegisterNodeRequest(BaseModel):
    node_id: str = Field(pattern=NODE_ID_PATTERN)
    display_name: str | None = None


class RegisterNodeResponse(BaseModel):
    node_id: str
    api_key: str


class RulesRequest(BaseModel):
    rules: list[dict[str, Any]]
