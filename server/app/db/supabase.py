"""Supabase implementation over its REST APIs (PostgREST + Storage) with httpx.

Uses the service_role key, which bypasses RLS. Never ship this key to a device.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx

from .base import BlobNotFound, Row

PAGE = 1000   # Supabase's default max rows per request

# Every bulk insert sends the same keys (PostgREST requires it).
LOG_COLUMNS = (
    "node_id", "uart_device_id", "device_seq", "event_kind", "act_id", "raw_input", "prompt", "model_output",
    "action", "model_version", "latency_ms", "status", "error_detail", "payload", "reward", "used_for_training",
)
EXAMPLE_COLUMNS = (
    "prompt", "context_json", "target_response", "source_type", "split", "source_log_id", "uart_device_id",
    "teacher", "quality", "dedupe_key",
)


class SupabaseError(RuntimeError):
    pass


def _check(resp: httpx.Response) -> httpx.Response:
    if resp.is_error:
        raise SupabaseError(f"{resp.request.method} {resp.request.url.path} -> {resp.status_code}: {resp.text[:300]}")
    return resp


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SupabaseRepository:
    def __init__(self, url: str, service_key: str, timeout: float = 30.0) -> None:
        self._http = httpx.Client(
            base_url=f"{url.rstrip('/')}/rest/v1",
            headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"},
            timeout=timeout,
        )

    # --- helpers ---
    def _select(self, table: str, params: dict[str, Any]) -> list[Row]:
        return _check(self._http.get(f"/{table}", params={"select": "*", **params})).json()

    def _select_all(self, table: str, params: dict[str, Any]) -> list[Row]:
        out: list[Row] = []
        offset = 0
        while True:
            page = self._select(table, {**params, "limit": PAGE, "offset": offset})
            out.extend(page)
            if len(page) < PAGE:
                return out
            offset += PAGE

    def _one(self, table: str, params: dict[str, Any]) -> Row | None:
        rows = self._select(table, {**params, "limit": 1})
        return rows[0] if rows else None

    def _insert(self, table: str, rows: list[Row] | Row, on_conflict: str | None = None, ignore_dupes: bool = False):
        prefer = ["return=representation"]
        if on_conflict:
            prefer.append("resolution=ignore-duplicates" if ignore_dupes else "resolution=merge-duplicates")
        params = {"on_conflict": on_conflict} if on_conflict else None
        return _check(self._http.post(f"/{table}", json=rows, params=params, headers={"Prefer": ",".join(prefer)})).json()

    def _patch(self, table: str, params: dict[str, Any], fields: Row) -> list[Row]:
        return _check(self._http.patch(f"/{table}", params=params, json=fields,
                                       headers={"Prefer": "return=representation"})).json()

    # --- edge_nodes ---
    def get_node(self, node_id: str) -> Row | None:
        return self._one("edge_nodes", {"id": f"eq.{node_id}"})

    def create_node(self, row: Row) -> Row:
        return self._insert("edge_nodes", row)[0]

    def update_node(self, node_id: str, fields: Row) -> None:
        self._patch("edge_nodes", {"id": f"eq.{node_id}"}, fields)

    # --- uart_devices ---
    def get_uart_device(self, node_id: str, device_key: str) -> Row | None:
        return self._one("uart_devices", {"node_id": f"eq.{node_id}", "device_key": f"eq.{device_key}"})

    def upsert_uart_device(self, node_id: str, device_key: str, fields: Row) -> Row:
        row = {"node_id": node_id, "device_key": device_key, **fields, "last_seen_at": _now()}
        return self._insert("uart_devices", row, on_conflict="node_id,device_key")[0]

    def list_uart_devices(self) -> list[Row]:
        return self._select_all("uart_devices", {"order": "first_seen_at.asc"})

    # --- interaction_logs ---
    def insert_logs(self, rows: list[Row]) -> int:
        if not rows:
            return 0
        uniform = [{c: r.get(c) for c in LOG_COLUMNS} for r in rows]
        for u in uniform:
            u["payload"] = u["payload"] or {}
            u["used_for_training"] = bool(u["used_for_training"])
        return len(self._insert("interaction_logs", uniform, on_conflict="node_id,device_seq", ignore_dupes=True))

    def fetch_untrained_interactions(self, limit: int) -> list[Row]:
        return self._select("interaction_logs", {"used_for_training": "is.false", "event_kind": "eq.interaction",
                                                 "order": "received_at.asc", "limit": limit})

    def count_untrained_interactions(self) -> int:
        resp = _check(self._http.get("/interaction_logs", params={
            "select": "id", "used_for_training": "is.false", "event_kind": "eq.interaction", "limit": 1,
        }, headers={"Prefer": "count=exact"}))
        total = resp.headers.get("content-range", "*/0").split("/")[-1]
        return int(total) if total.isdigit() else 0

    def mark_logs_used(self, ids: list[int]) -> None:
        for i in range(0, len(ids), 200):
            chunk = ",".join(str(x) for x in ids[i:i + 200])
            self._patch("interaction_logs", {"id": f"in.({chunk})"}, {"used_for_training": True})

    def set_reward_by_act(self, node_id: str, act_id: int, reward: float) -> int:
        return len(self._patch("interaction_logs", {"node_id": f"eq.{node_id}", "act_id": f"eq.{act_id}",
                                                    "event_kind": "eq.interaction"}, {"reward": reward}))

    # --- training_datasets ---
    def insert_examples(self, rows: list[Row]) -> int:
        if not rows:
            return 0
        uniform = [{c: r.get(c) for c in EXAMPLE_COLUMNS} for r in rows]
        for u in uniform:
            u["context_json"] = u["context_json"] or {}
        return len(self._insert("training_datasets", uniform, on_conflict="dedupe_key", ignore_dupes=True))

    def list_examples(self, split: str | None = None) -> list[Row]:
        params: dict[str, Any] = {"order": "id.asc"}
        if split:
            params["split"] = f"eq.{split}"
        return self._select_all("training_datasets", params)

    # --- model_versions ---
    def get_active_model(self) -> Row | None:
        return self._one("model_versions", {"is_active": "is.true"})

    def get_model(self, version: int) -> Row | None:
        return self._one("model_versions", {"version": f"eq.{version}"})

    def get_model_by_sha(self, sha256: str) -> Row | None:
        return self._one("model_versions", {"sha256": f"eq.{sha256}"})

    def latest_model_version(self) -> int:
        row = self._one("model_versions", {"order": "version.desc"})
        return int(row["version"]) if row else 0

    def insert_model(self, row: Row) -> Row:
        return self._insert("model_versions", row)[0]

    def activate_model(self, version: int) -> None:
        _check(self._http.post("/rpc/activate_model_version", json={"p_version": version}))

    # --- evolution_runs ---
    def create_run(self, row: Row) -> Row:
        return self._insert("evolution_runs", row)[0]

    def update_run(self, run_id: str, fields: Row) -> None:
        self._patch("evolution_runs", {"id": f"eq.{run_id}"}, fields)

    def get_run(self, run_id: str) -> Row | None:
        return self._one("evolution_runs", {"id": f"eq.{run_id}"})

    def list_runs(self, limit: int = 20) -> list[Row]:
        return self._select("evolution_runs", {"order": "created_at.desc", "limit": limit})

    def next_queued_run(self) -> Row | None:
        return self._one("evolution_runs", {"status": "eq.queued", "order": "created_at.asc"})


class SupabaseBlobStore:
    def __init__(self, url: str, service_key: str, timeout: float = 120.0) -> None:
        self._base = f"{url.rstrip('/')}/storage/v1"
        self._http = httpx.Client(headers={"apikey": service_key, "Authorization": f"Bearer {service_key}"},
                                  timeout=timeout)

    @staticmethod
    def _p(bucket: str, path: str) -> str:
        return f"{quote(bucket)}/{quote(path)}"

    def put(self, bucket: str, path: str, data: bytes) -> None:
        _check(self._http.post(f"{self._base}/object/{self._p(bucket, path)}", content=data,
                               headers={"Content-Type": "application/octet-stream", "x-upsert": "true"}))

    def get(self, bucket: str, path: str) -> bytes:
        resp = self._http.get(f"{self._base}/object/{self._p(bucket, path)}")
        if resp.status_code in (400, 404):
            raise BlobNotFound(f"{bucket}/{path}")
        return _check(resp).content

    def signed_url(self, bucket: str, path: str, expires_s: int = 3600) -> str:
        resp = _check(self._http.post(f"{self._base}/object/sign/{self._p(bucket, path)}", json={"expiresIn": expires_s}))
        signed = resp.json().get("signedURL") or resp.json().get("signedUrl")
        if not signed:
            raise SupabaseError(f"no signed URL in response: {resp.text[:200]}")
        return signed if signed.startswith("http") else f"{self._base}{signed}"
