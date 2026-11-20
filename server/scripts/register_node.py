"""Register an edge node and print the lines for edge/secrets.h.

  python scripts/register_node.py --server http://localhost:8000 --node-id edge-01
  (ADMIN_TOKEN env var or --admin-token; dev mode default is "dev-admin-token")
"""

from __future__ import annotations

import argparse
import os

import httpx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("--node-id", required=True)
    p.add_argument("--name")
    p.add_argument("--admin-token", default=os.environ.get("ADMIN_TOKEN", "dev-admin-token"))
    args = p.parse_args()

    r = httpx.post(f"{args.server.rstrip('/')}/api/v1/admin/nodes",
                   json={"node_id": args.node_id, "display_name": args.name},
                   headers={"Authorization": f"Bearer {args.admin_token}"}, timeout=30)
    if r.is_error:
        raise SystemExit(f"{r.status_code}: {r.text}")
    key = r.json()["api_key"]
    print("// edge/secrets.h  (the key is shown once; the server keeps only its hash)")
    print(f'#define EDGE_NODE_ID         "{args.node_id}"')
    print(f'#define EDGE_NODE_API_KEY    "{key}"')
    print(f'#define EDGE_SERVER_BASE_URL "{args.server.rstrip("/")}"')


if __name__ == "__main__":
    main()
