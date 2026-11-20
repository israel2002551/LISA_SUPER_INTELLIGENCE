"""Create or inspect the ECDSA P-256 key that signs OTA manifests.

  python scripts/gen_signing_key.py                      new key -> prints the PEM for OTA_SIGNING_KEY_PEM
  python scripts/gen_signing_key.py --out key.pem        new key -> file
  python scripts/gen_signing_key.py --show-public key.pem
        prints EDGE_OTA_PUBLIC_KEY_PEM for edge/secrets.h

Keep the private key out of git (*.pem is ignored). Anyone holding it can push
models to every device.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evolution.signing import ManifestSigner  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, help="write the new private key here instead of printing it")
    p.add_argument("--show-public", type=Path, metavar="KEY_PEM", help="print the C public key for an existing key")
    args = p.parse_args()

    if args.show_public:
        signer = ManifestSigner.from_pem(args.show_public.read_bytes())
    else:
        signer = ManifestSigner.generate()
        if args.out:
            args.out.write_bytes(signer.private_pem())
            print(f"Private key written to {args.out}")
        else:
            print("# Private key (Render env var OTA_SIGNING_KEY_PEM):")
            print(signer.private_pem().decode())
    print("// edge/secrets.h")
    print(f"#define EDGE_OTA_PUBLIC_KEY_PEM \\\n    {signer.public_pem_c_string()}")


if __name__ == "__main__":
    main()
