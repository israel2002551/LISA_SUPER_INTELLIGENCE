"""Operator endpoints (ADMIN_TOKEN): evolution runs, nodes, models, device rules."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from ..evolution.model_spec import SPEC
from ..evolution.pipeline import TOKENIZER_BLOB, bootstrap_model, create_run, run_evolution
from ..evolution.tokenizer import BpeTokenizer
from ..schemas import RegisterNodeRequest, RegisterNodeResponse, RulesRequest, TriggerRequest, TriggerResponse
from ..security import Admin, Ctx, hash_key, new_api_key

router = APIRouter(tags=["admin"])


# ---- evolution ----
@router.post("/evolution/trigger", response_model=TriggerResponse)
def trigger(req: TriggerRequest, _: Admin, ctx: Ctx, background: BackgroundTasks) -> TriggerResponse:
    params = {k: v for k, v in req.model_dump().items() if v not in (None, False)}
    run = create_run(ctx, "manual", params)
    if ctx.settings.evolution_inline:
        background.add_task(run_evolution, ctx, run["id"])
        return TriggerResponse(run_id=run["id"], status="queued", executor="inline")
    return TriggerResponse(run_id=run["id"], status="queued", executor="worker")


@router.get("/evolution/runs")
def list_runs(_: Admin, ctx: Ctx, limit: int = 20) -> list[dict[str, Any]]:
    return ctx.repo.list_runs(min(limit, 200))


@router.get("/evolution/runs/{run_id}")
def get_run(run_id: str, _: Admin, ctx: Ctx) -> dict[str, Any]:
    run = ctx.repo.get_run(run_id)
    if not run:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown run")
    return run


# ---- nodes ----
@router.post("/admin/nodes", response_model=RegisterNodeResponse)
def register_node(req: RegisterNodeRequest, _: Admin, ctx: Ctx) -> RegisterNodeResponse:
    if ctx.repo.get_node(req.node_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "node already registered")
    key = new_api_key()
    ctx.repo.create_node({"id": req.node_id, "display_name": req.display_name, "api_key_hash": hash_key(key)})
    return RegisterNodeResponse(node_id=req.node_id, api_key=key)


# ---- models ----
@router.post("/admin/models/bootstrap")
async def bootstrap(request: Request, _: Admin, ctx: Ctx, version: int = 1) -> dict[str, Any]:
    """Body: a model image (application/octet-stream), e.g. from gemma4_slice_int16.py.
    Registers it as `version` and makes it active."""
    image = await request.body()
    try:
        model = bootstrap_model(ctx, image, version)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    return {"version": model["version"], "size_bytes": model["size_bytes"], "sha256": model["sha256"],
            "storage_path": model["storage_path"], "active": True}


@router.put("/admin/assets/tokenizer")
async def upload_tokenizer(request: Request, _: Admin, ctx: Ctx) -> dict[str, Any]:
    """Body: tokenizer.bin. Stored at models/assets/tokenizer.bin for workers that
    don't have the file in their checkout."""
    data = await request.body()
    try:
        BpeTokenizer(data, SPEC.vocab)
    except Exception as e:  # noqa: BLE001 - any parse failure means a bad upload
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"not a llama2.c tokenizer.bin: {e}") from e
    ctx.blobs.put(*TOKENIZER_BLOB, data)
    return {"stored": "/".join(TOKENIZER_BLOB), "bytes": len(data)}


@router.post("/admin/models/{version}/activate")
def activate(version: int, _: Admin, ctx: Ctx) -> dict[str, Any]:
    """Manual rollout or rollback: devices converge to this version on their next OTA check."""
    if not ctx.repo.get_model(version):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown model version")
    ctx.repo.activate_model(version)
    return {"active_version": version}


# ---- device rules (operator teaching) ----
@router.put("/admin/devices/{node_id}/{device_key}/rules")
def set_rules(node_id: str, device_key: str, req: RulesRequest, _: Admin, ctx: Ctx) -> dict[str, Any]:
    if not ctx.repo.get_uart_device(node_id, device_key):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown device")
    row = ctx.repo.upsert_uart_device(node_id, device_key, {"learned_rules": req.rules})
    return {"device_key": device_key, "rules": row["learned_rules"]}


@router.get("/admin/devices")
def list_devices(_: Admin, ctx: Ctx) -> list[dict[str, Any]]:
    return ctx.repo.list_uart_devices()
