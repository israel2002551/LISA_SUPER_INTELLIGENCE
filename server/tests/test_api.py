"""HTTP API with the in-memory backend: what the ESP32 and an operator see."""

from __future__ import annotations

import base64
import hashlib

import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

from app.evolution.image_codec import set_model_version
from app.evolution.signing import manifest_message

from .conftest import ADMIN, device_event, interaction_event, register, status_block


def telemetry(client, headers, events, sha="", version=0, devices=None):
    return client.post("/api/v1/telemetry", headers=headers, json={
        "node_id": headers["X-Node-Id"], "fw_version": "edge-0.1.0", "status": status_block(sha, version),
        "devices": devices or [], "events": events})


def test_root_and_health(client):
    assert client.get("/").json()["name"] == "LISA Evolution Server"
    assert client.head("/").status_code == 200          # hosting health probes use HEAD /
    h = client.get("/healthz").json()
    assert h["ok"] and h["ai"] == "LISA"


def test_auth(client):
    headers = register(client)
    assert client.post("/api/v1/telemetry", json={"node_id": "edge-01"}).status_code == 401
    bad = {**headers, "Authorization": "Bearer wrong"}
    assert telemetry(client, bad, []).status_code == 401
    assert client.post("/api/v1/admin/nodes", json={"node_id": "x"}, headers={"Authorization": "Bearer no"}).status_code == 401
    assert client.post("/api/v1/admin/nodes", json={"node_id": "edge-01"}, headers=ADMIN).status_code == 409
    other = register(client, "edge-02")
    r = client.post("/api/v1/telemetry", headers=other, json={"node_id": "edge-01", "events": []})
    assert r.status_code == 403


def test_telemetry_ingestion_is_idempotent_and_scores_rewards(client, ctx):
    headers = register(client)
    events = [device_event(1),
              interaction_event(2, 31.0, '{"cmd":"set_fan","args":{"level":2}}', "ok"),
              interaction_event(3, 22.0, "garbage", "invalid_output"),
              interaction_event(4, 10.0, "NOP", "nop"),
              {"seq": 5, "type": 6, "data": {"kind": "feedback", "dev": "thermo-01", "ref_act_id": 4, "value": "bad"}}]
    r = telemetry(client, headers, events)
    assert r.status_code == 200, r.text
    assert r.json() == {"acked_seq": 5, "inserted": 5, "rediscover": False}
    assert telemetry(client, headers, events).json()["inserted"] == 0          # re-upload: no duplicates

    logs = {l["device_seq"]: l for l in ctx.repo.logs}
    assert logs[2]["reward"] == 1.0 and logs[3]["reward"] == 0.0
    assert logs[4]["reward"] == 0.0                                            # set by the "bad" feedback
    dev = ctx.repo.get_uart_device("edge-01", "thermo-01")
    assert dev["identified"] and dev["bus"] == "uart_fast" and dev["inferred_intent"] == "alerts"
    assert dev["capability_manifest"]["accepts"][0]["cmd"] == "set_fan"
    node = ctx.repo.get_node("edge-01")
    assert node["w25q_jedec_id"] == "ef4018" and node["last_heartbeat_at"]


def test_raw_device_and_classify(client, ctx):
    headers = register(client)
    nmea = b"$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47\r\n"
    ev = {"seq": 1, "type": 2, "data": {"kind": "device", "dev": "raw:uart1", "identified": False, "baud": 9600,
                                       "format": "nmea", "type": "unknown", "sample_hex": nmea.hex()}}
    assert telemetry(client, headers, [ev]).status_code == 200
    assert ctx.repo.get_uart_device("edge-01", "raw:uart1")["device_type"] == "gps"
    r = client.post("/api/v1/uart/classify", headers=headers, json={
        "node_id": "edge-01", "device_key": "raw:uart1", "baud": 9600, "format_guess": "nmea", "sample_hex": nmea.hex()})
    assert r.status_code == 200
    assert r.json()["format"] == "nmea" and r.json()["device_type"] == "gps" and r.json()["source"] == "heuristic"


def test_ota_check_full_update_with_valid_signature(client, ctx, base_image, signer):
    headers = register(client)
    q = {"node_id": "edge-01", "current_version": 0, "current_sha256": "0" * 64}
    assert client.get("/api/v1/ota/check", params=q, headers=headers).json()["update"] is False   # no model yet

    r = client.post("/api/v1/admin/models/bootstrap", params={"version": 1}, content=base_image,
                    headers={**ADMIN, "Content-Type": "application/octet-stream"})
    assert r.status_code == 200, r.text
    v1 = set_model_version(base_image, 1)
    assert r.json()["sha256"] == hashlib.sha256(v1).hexdigest()

    m = client.get("/api/v1/ota/check", params=q, headers=headers).json()
    assert m["update"] and m["kind"] == "full" and m["version"] == 1 and m["size"] == len(v1)
    # The device verifies exactly this with mbedtls (ECDSA P-256 over SHA-256 of the message).
    signer._key.public_key().verify(base64.b64decode(m["signature"]),
                                    manifest_message(1, len(v1), m["sha256"]), ec.ECDSA(hashes.SHA256()))
    # Signed download URL (dev mode serves it from local disk)
    path = httpx.URL(m["url"])
    dl = client.get(path.path, params=dict(path.params))
    assert dl.status_code == 200 and dl.content == v1
    tampered = dict(path.params, sig="0" * 64)
    assert client.get(path.path, params=tampered).status_code == 403

    same = client.get("/api/v1/ota/check", headers=headers,
                      params={"node_id": "edge-01", "current_version": 1, "current_sha256": m["sha256"]}).json()
    assert same["update"] is False
