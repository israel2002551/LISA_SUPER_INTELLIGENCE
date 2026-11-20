"""FastAPI app factory.

Run locally (dev mode, no Supabase):   uvicorn app.main:create_app --factory --reload
On Render:                             uvicorn app.main:create_app --factory --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import FastAPI

from .config import Settings, get_settings
from .db.memory import InMemoryRepository, LocalBlobStore
from .db.supabase import SupabaseBlobStore, SupabaseRepository
from .evolution.pipeline import EvolutionContext, make_teacher
from .evolution.signing import ManifestSigner
from .routers import admin, blobs, device

log = logging.getLogger(__name__)


def make_signer(s: Settings) -> ManifestSigner:
    if s.ota_signing_key_pem:
        pem = s.ota_signing_key_pem.get_secret_value().strip()
        if "\n" not in pem:          # pasted into a dashboard as one line with literal "\n"
            pem = pem.replace("\\n", "\n")
        return ManifestSigner.from_pem(pem.encode() + b"\n")
    if s.ota_signing_key_path:
        return ManifestSigner.from_pem(s.ota_signing_key_path.read_bytes())
    if s.dev_mode:
        return ManifestSigner.load_or_create(s.data_dir / "ota_signing_key.pem")
    raise ValueError("Set OTA_SIGNING_KEY_PEM (see server/scripts/gen_signing_key.py)")


def build_context(s: Settings) -> EvolutionContext:
    if s.dev_mode:
        log.warning("DEV MODE: no SUPABASE_URL; using an in-memory database and %s for files", s.data_dir)
        secret = hashlib.sha256(b"blobs|" + s.admin_token.get_secret_value().encode()).digest()
        repo, blob_store = InMemoryRepository(), LocalBlobStore(s.data_dir / "blobs", s.public_base_url, secret)
    else:
        key = s.supabase_service_role_key.get_secret_value()   # validated by Settings
        repo, blob_store = SupabaseRepository(s.supabase_url, key), SupabaseBlobStore(s.supabase_url, key)
    return EvolutionContext(settings=s, repo=repo, blobs=blob_store, signer=make_signer(s), teacher=make_teacher(s))


def create_app(ctx: EvolutionContext | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    ctx = ctx or build_context(get_settings())
    name = ctx.settings.ai_name
    app = FastAPI(title=f"{name} Evolution Server", version="0.1.0",
                  description=f"Telemetry, OTA and continual learning for {name}'s ESP32-S3 + W25Q edge nodes.")
    app.state.ctx = ctx
    app.include_router(device.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")
    if app.state.ctx.settings.dev_mode:
        app.include_router(blobs.router, prefix="/api/v1")

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        c = app.state.ctx
        active = c.repo.get_active_model()
        return {"ok": True, "ai": c.settings.ai_name, "dev_mode": c.settings.dev_mode, "trainer": c.settings.trainer,
                "teacher": c.settings.teacher, "active_model": active["version"] if active else None}

    return app
