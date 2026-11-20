"""In-memory repository and local-disk blob store.

Used by the tests and by dev mode (no SUPABASE_URL): the whole system can run
on a laptop, including OTA downloads to a real ESP32 on the same network.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .base import BlobNotFound, Row


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryRepository:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.nodes: dict[str, Row] = {}
        self.devices: dict[str, Row] = {}           # id -> row
        self.logs: list[Row] = []
        self._log_keys: set[tuple[str, int]] = set()
        self.examples: list[Row] = []
        self._example_keys: set[str] = set()
        self.models: dict[int, Row] = {}
        self.runs: dict[str, Row] = {}
        self._next_log_id = 1
        self._next_example_id = 1

    # --- edge_nodes ---
    def get_node(self, node_id: str) -> Row | None:
        with self._lock:
            row = self.nodes.get(node_id)
            return copy.deepcopy(row) if row else None

    def create_node(self, row: Row) -> Row:
        with self._lock:
            if row["id"] in self.nodes:
                raise ValueError(f"node {row['id']} already exists")
            full = {"partition_state": {}, "status": {}, "created_at": _now(), "updated_at": _now(), **row}
            self.nodes[row["id"]] = full
            return copy.deepcopy(full)

    def update_node(self, node_id: str, fields: Row) -> None:
        with self._lock:
            self.nodes[node_id].update(copy.deepcopy(fields), updated_at=_now())

    # --- uart_devices ---
    def get_uart_device(self, node_id: str, device_key: str) -> Row | None:
        with self._lock:
            for d in self.devices.values():
                if d["node_id"] == node_id and d["device_key"] == device_key:
                    return copy.deepcopy(d)
            return None

    def upsert_uart_device(self, node_id: str, device_key: str, fields: Row) -> Row:
        with self._lock:
            for d in self.devices.values():
                if d["node_id"] == node_id and d["device_key"] == device_key:
                    d.update(copy.deepcopy(fields), last_seen_at=_now())
                    return copy.deepcopy(d)
            row = {
                "id": str(uuid.uuid4()), "node_id": node_id, "device_key": device_key, "port": 1,
                "identified": False, "baud": None, "bus": "uart", "data_format": "unknown", "device_type": None,
                "capability_manifest": {}, "inferred_intent": None, "protocol_schema": {}, "learned_rules": [],
                "first_seen_at": _now(), "last_seen_at": _now(),
            }
            row.update(copy.deepcopy(fields))
            self.devices[row["id"]] = row
            return copy.deepcopy(row)

    def list_uart_devices(self) -> list[Row]:
        with self._lock:
            return copy.deepcopy(list(self.devices.values()))

    # --- interaction_logs ---
    def insert_logs(self, rows: list[Row]) -> int:
        inserted = 0
        with self._lock:
            for r in rows:
                key = (r["node_id"], int(r["device_seq"]))
                if key in self._log_keys:
                    continue
                self._log_keys.add(key)
                full = {"id": self._next_log_id, "used_for_training": False, "received_at": _now(), **copy.deepcopy(r)}
                self._next_log_id += 1
                self.logs.append(full)
                inserted += 1
        return inserted

    def fetch_untrained_interactions(self, limit: int) -> list[Row]:
        with self._lock:
            out = [l for l in self.logs if not l["used_for_training"] and l["event_kind"] == "interaction"]
            return copy.deepcopy(out[:limit])

    def count_untrained_interactions(self) -> int:
        with self._lock:
            return sum(1 for l in self.logs if not l["used_for_training"] and l["event_kind"] == "interaction")

    def mark_logs_used(self, ids: list[int]) -> None:
        wanted = set(ids)
        with self._lock:
            for l in self.logs:
                if l["id"] in wanted:
                    l["used_for_training"] = True

    def set_reward_by_act(self, node_id: str, act_id: int, reward: float) -> int:
        n = 0
        with self._lock:
            for l in self.logs:
                if l["node_id"] == node_id and l.get("act_id") == act_id and l["event_kind"] == "interaction":
                    l["reward"] = reward
                    n += 1
        return n

    # --- training_datasets ---
    def insert_examples(self, rows: list[Row]) -> int:
        inserted = 0
        with self._lock:
            for r in rows:
                if r["dedupe_key"] in self._example_keys:
                    continue
                self._example_keys.add(r["dedupe_key"])
                self.examples.append({"id": self._next_example_id, "created_at": _now(), **copy.deepcopy(r)})
                self._next_example_id += 1
                inserted += 1
        return inserted

    def list_examples(self, split: str | None = None) -> list[Row]:
        with self._lock:
            return copy.deepcopy([e for e in self.examples if split is None or e["split"] == split])

    # --- model_versions ---
    def get_active_model(self) -> Row | None:
        with self._lock:
            for m in self.models.values():
                if m["is_active"]:
                    return copy.deepcopy(m)
            return None

    def get_model(self, version: int) -> Row | None:
        with self._lock:
            m = self.models.get(version)
            return copy.deepcopy(m) if m else None

    def get_model_by_sha(self, sha256: str) -> Row | None:
        with self._lock:
            for m in self.models.values():
                if m["sha256"] == sha256:
                    return copy.deepcopy(m)
            return None

    def latest_model_version(self) -> int:
        with self._lock:
            return max(self.models, default=0)

    def insert_model(self, row: Row) -> Row:
        with self._lock:
            if row["version"] in self.models:
                raise ValueError(f"model version {row['version']} exists")
            full = {"is_active": False, "metrics_json": {}, "created_at": _now(), **copy.deepcopy(row)}
            self.models[row["version"]] = full
            return copy.deepcopy(full)

    def activate_model(self, version: int) -> None:
        with self._lock:
            if version not in self.models:
                raise ValueError(f"unknown model version {version}")
            for v, m in self.models.items():
                m["is_active"] = v == version

    # --- evolution_runs ---
    def create_run(self, row: Row) -> Row:
        with self._lock:
            full = {"id": str(uuid.uuid4()), "status": "queued", "params": {}, "stats": {}, "error": None,
                    "produced_version": None, "created_at": _now(), "started_at": None, "finished_at": None,
                    **copy.deepcopy(row)}
            self.runs[full["id"]] = full
            return copy.deepcopy(full)

    def update_run(self, run_id: str, fields: Row) -> None:
        with self._lock:
            self.runs[run_id].update(copy.deepcopy(fields))

    def get_run(self, run_id: str) -> Row | None:
        with self._lock:
            r = self.runs.get(run_id)
            return copy.deepcopy(r) if r else None

    def list_runs(self, limit: int = 20) -> list[Row]:
        with self._lock:
            return copy.deepcopy(sorted(self.runs.values(), key=lambda r: r["created_at"], reverse=True)[:limit])

    def next_queued_run(self) -> Row | None:
        with self._lock:
            queued = sorted((r for r in self.runs.values() if r["status"] == "queued"), key=lambda r: r["created_at"])
            return copy.deepcopy(queued[0]) if queued else None


class LocalBlobStore:
    """Files under <root>/<bucket>/<path>. Signed URLs point at this server's
    /api/v1/blobs route and carry an HMAC so only issued URLs work."""

    def __init__(self, root: Path, public_base_url: str, secret: bytes) -> None:
        self.root = Path(root)
        self.public_base_url = public_base_url.rstrip("/")
        self._secret = secret

    def _file(self, bucket: str, path: str) -> Path:
        p = (self.root / bucket / path).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("path escapes the blob root")
        return p

    def put(self, bucket: str, path: str, data: bytes) -> None:
        f = self._file(bucket, path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)

    def get(self, bucket: str, path: str) -> bytes:
        f = self._file(bucket, path)
        if not f.is_file():
            raise BlobNotFound(f"{bucket}/{path}")
        return f.read_bytes()

    def file_path(self, bucket: str, path: str) -> Path:
        return self._file(bucket, path)

    def _sign(self, bucket: str, path: str, exp: int) -> str:
        msg = f"{bucket}/{path}|{exp}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def signed_url(self, bucket: str, path: str, expires_s: int = 3600) -> str:
        exp = int(time.time()) + expires_s
        return f"{self.public_base_url}/api/v1/blobs/{bucket}/{path}?exp={exp}&sig={self._sign(bucket, path, exp)}"

    def verify(self, bucket: str, path: str, exp: int, sig: str) -> bool:
        return exp >= time.time() and hmac.compare_digest(sig, self._sign(bucket, path, exp))
