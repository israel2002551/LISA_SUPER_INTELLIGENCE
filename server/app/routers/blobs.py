"""Dev mode only: serves model files from local disk via HMAC-signed URLs, so a
real ESP32 on the LAN can download OTA images from a laptop running the server."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from ..db.memory import LocalBlobStore
from ..security import Ctx

router = APIRouter(tags=["dev"])


@router.get("/blobs/{bucket}/{path:path}")
def download(bucket: str, path: str, exp: int, sig: str, ctx: Ctx) -> FileResponse:
    blobs = ctx.blobs
    if not isinstance(blobs, LocalBlobStore) or not blobs.verify(bucket, path, exp, sig):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid or expired link")
    f = blobs.file_path(bucket, path)
    if not f.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such object")
    return FileResponse(f, media_type="application/octet-stream")
