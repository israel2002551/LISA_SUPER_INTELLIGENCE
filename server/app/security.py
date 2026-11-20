"""Auth: per-node API keys (devices) and an admin token (operators, CI).

Node keys are random 256-bit strings; only their SHA-256 is stored in
edge_nodes.api_key_hash. Devices send:
  Authorization: Bearer <node api key>
  X-Node-Id: <node id>
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Request, status

from .evolution.pipeline import EvolutionContext


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_api_key() -> str:
    return secrets.token_urlsafe(32)


def get_ctx(request: Request) -> EvolutionContext:
    return request.app.state.ctx


Ctx = Annotated[EvolutionContext, Depends(get_ctx)]


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    return authorization[len("Bearer "):].strip()


def require_node(ctx: Ctx, authorization: Annotated[str | None, Header()] = None,
                 x_node_id: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    key = _bearer(authorization)
    node = ctx.repo.get_node(x_node_id) if x_node_id else None
    if node is None or not hmac.compare_digest(node["api_key_hash"], hash_key(key)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown node or bad key")
    return node


def require_admin(ctx: Ctx, authorization: Annotated[str | None, Header()] = None) -> None:
    expected = ctx.settings.admin_token.get_secret_value()
    if not hmac.compare_digest(_bearer(authorization), expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad admin token")


Node = Annotated[dict[str, Any], Depends(require_node)]
Admin = Annotated[None, Depends(require_admin)]
