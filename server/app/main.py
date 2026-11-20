"""FastAPI app factory.

Run locally (dev mode, no Supabase):   uvicorn app.main:create_app --factory --reload
On Render:                             uvicorn app.main:create_app --factory --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import hashlib
import logging
import re

from fastapi import FastAPI

from .config import Settings, get_settings
from .db.memory import InMemoryRepository, LocalBlobStore
from .db.supabase import SupabaseBlobStore, SupabaseRepository
from .evolution.pipeline import EvolutionContext, make_teacher
from .evolution.signing import ManifestSigner
from .routers import admin, blobs, device

log = logging.getLogger(__name__)


def normalize_pem(value: str) -> bytes:
    """Rebuilds a PEM that went through a dashboard field: line breaks kept, written
    as literal "\\n", turned into spaces, or stripped entirely."""
    v = value.strip().replace("\\n", "\n").replace("\r\n", "\n")
    m = re.fullmatch(r"-----BEGIN ([A-Z0-9 ]+)-----(.*?)-----END \1-----", v, re.S)
    if not m:
        return v.encode()
    body = re.sub(r"\s+", "", m.group(2))
    lines = [body[i:i + 64] for i in range(0, len(body), 64)]
    label = m.group(1)
    return f"-----BEGIN {label}-----\n{chr(10).join(lines)}\n-----END {label}-----\n".encode()


def make_signer(s: Settings) -> ManifestSigner:
    if s.ota_signing_key_pem:
        return ManifestSigner.from_pem(normalize_pem(s.ota_signing_key_pem.get_secret_value()))
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

    @app.api_route("/", methods=["GET", "HEAD"], tags=["ops"])
    def root() -> dict:
        return {"name": f"{name} Evolution Server", "status": "running", "health": "/healthz", "api_docs": "/docs"}

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict:
        c = app.state.ctx
        active = c.repo.get_active_model()
        return {"ok": True, "ai": c.settings.ai_name, "dev_mode": c.settings.dev_mode, "trainer": c.settings.trainer,
                "teacher": c.settings.teacher, "active_model": active["version"] if active else None}

    return app
