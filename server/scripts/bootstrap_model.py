"""Register the first model version and write the matching file to provision a device.

  python scripts/bootstrap_model.py --server http://localhost:8000 \
      --image ../cluster_binaries/gemma4_w25q128_int16.bin --version 1

Writes <image>.v1.w25img: flash THAT file over USB
(python gemma4/w25q128_upload.py COM3 <file>) so the device's SHA-256 matches
model v1 and it doesn't immediately download the same model again.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evolution.image_codec import set_model_version  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--tokenizer", type=Path, default=Path(__file__).resolve().parents[2] / "tokenizer.bin")
    p.add_argument("--admin-token", default=os.environ.get("ADMIN_TOKEN", "dev-admin-token"))
    args = p.parse_args()

    versioned = set_model_version(args.image.read_bytes(), args.version)
    out = args.image.with_suffix(f".v{args.version}.w25img")
    out.write_bytes(versioned)
    auth = {"Authorization": f"Bearer {args.admin_token}"}

    if args.tokenizer.is_file():   # so a worker without tokenizer.bin in its checkout can still train
        r = httpx.put(f"{args.server.rstrip('/')}/api/v1/admin/assets/tokenizer", content=args.tokenizer.read_bytes(),
                      headers={**auth, "Content-Type": "application/octet-stream"}, timeout=120)
        if r.is_error:
            raise SystemExit(f"tokenizer upload failed: {r.status_code}: {r.text}")
        print(f"Uploaded {args.tokenizer.name} to storage ({r.json()['bytes']} bytes).")

    r = httpx.post(f"{args.server.rstrip('/')}/api/v1/admin/models/bootstrap", params={"version": args.version},
                   content=versioned, headers={"Authorization": f"Bearer {args.admin_token}",
                                               "Content-Type": "application/octet-stream"}, timeout=300)
    if r.is_error:
        raise SystemExit(f"{r.status_code}: {r.text}")
    model = r.json()
    assert model["sha256"] == hashlib.sha256(versioned).hexdigest(), "server stored different bytes"
    print(f"Model v{model['version']} registered and active (sha256 {model['sha256'][:16]}...).")
    print(f"Provision devices with: {out}")


if __name__ == "__main__":
    main()
