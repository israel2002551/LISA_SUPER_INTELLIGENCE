"""The evolution loop end to end with the mock trainer (no ML needed)."""

from __future__ import annotations

import httpx

from app.evolution.curator import curate
from app.evolution.delta import apply_delta
from app.evolution.pipeline import bootstrap_model, create_run, gate, run_evolution, should_trigger

from .conftest import device_event, interaction_event, register, status_block


def _ingest(client, headers, events):
    r = client.post("/api/v1/telemetry", headers=headers, json={
        "node_id": "edge-01", "status": status_block(), "events": events})
    assert r.status_code == 200, r.text


def test_curation_turns_outcomes_into_examples(client, ctx):
    headers = register(client)
    _ingest(client, headers, [
        device_event(1),
        interaction_event(2, 31.0, '{"cmd":"set_fan","args":{"level":2}}', "ok"),    # right, acked -> real_log
        interaction_event(3, 22.0, '{"cmd":"set_fan","args":{"level":3}}', "ok"),    # no rule covers 22 C: the ack teaches
        interaction_event(4, 35.0, "bad json", "invalid_output"),                     # rule matches -> failure_recovery
        interaction_event(5, 20.0, "NOP", "nop"),                                     # agrees with the rules
        interaction_event(6, 20.0, "", "error"),                                      # no model: skipped
        interaction_event(7, 33.0, '{"cmd":"set_fan","args":{"level":1}}', "ok"),    # acked, but a rule says level 2
    ])
    stats = curate(ctx.repo, ctx.teacher, 100)
    assert stats == {"logs": 6, "real_log": 3, "failure_recovery": 2, "skipped": 1, "inserted": 5}
    targets = {(e["source_type"], e["target_response"]) for e in ctx.repo.examples}
    assert ("real_log", '{"cmd":"set_fan","args":{"level":3}}') in targets
    assert ("failure_recovery", '{"cmd":"set_fan","args":{"level":2}}') in targets
    assert ("real_log", '{"cmd":"set_fan","args":{"level":1}}') not in targets
    assert ctx.repo.count_untrained_interactions() == 0          # all marked used


def test_gate():
    assert gate({"exact_match": 0.2, "val_loss": 2.0}, {"exact_match": 0.5, "val_loss": 1.0}, 0.0, False)[0]
    assert not gate({"exact_match": 0.5, "val_loss": 1.0}, {"exact_match": 0.4, "val_loss": 0.5}, 0.0, False)[0]
    assert gate({"exact_match": 0.5, "val_loss": 1.0}, {"exact_match": 0.5, "val_loss": 0.9}, 0.0, False)[0]
    assert not gate({"mock": True}, {"mock": True}, 0.0, False)[0]
    assert gate({"mock": True}, {"mock": True}, 0.0, True)[0]


def test_mock_run_publishes_v2_with_delta_and_ota_serves_it(client, ctx, base_image):
    headers = register(client)
    v1 = bootstrap_model(ctx, base_image, 1)
    _ingest(client, headers, [device_event(1)] + [
        interaction_event(10 + i, 20.0 + i, "NOP", "nop") for i in range(5)])

    rejected = run_evolution(ctx, create_run(ctx, "manual")["id"])
    assert rejected["status"] == "rejected" and "mock" in rejected["error"]

    run = run_evolution(ctx, create_run(ctx, "manual", {"force_publish": True})["id"])
    assert run["status"] == "published", run
    assert run["produced_version"] == 2 and run["stats"]["synthetic_inserted"] > 0
    v2 = ctx.repo.get_model(2)
    assert v2["is_active"] and v2["parent_version"] == 1 and v2["delta_from_version"] == 1

    m = client.get("/api/v1/ota/check", headers=headers,
                   params={"node_id": "edge-01", "current_version": 1, "current_sha256": v1["sha256"]}).json()
    assert m["update"] and m["kind"] == "delta" and m["version"] == 2
    url = httpx.URL(m["url"])
    delta = client.get(url.path, params=dict(url.params)).content
    base = ctx.blobs.get("models", v1["storage_path"])
    assert apply_delta(base, delta) == ctx.blobs.get("models", v2["storage_path"])

    # A device on an unknown image gets the full file instead.
    full = client.get("/api/v1/ota/check", headers=headers,
                      params={"node_id": "edge-01", "current_sha256": "f" * 64}).json()
    assert full["kind"] == "full"

    # Operator rollback: re-activate v1; devices on v2 are told to install v1.
    client.post("/api/v1/admin/models/1/activate", headers={"Authorization": "Bearer dev-admin-token"})
    back = client.get("/api/v1/ota/check", headers=headers,
                      params={"node_id": "edge-01", "current_sha256": v2["sha256"]}).json()
    assert back["update"] and back["version"] == 1


def test_trigger_endpoint_and_worker_policy(client, ctx):
    r = client.post("/api/v1/evolution/trigger", json={"force_publish": True},
                    headers={"Authorization": "Bearer dev-admin-token"})
    assert r.status_code == 200 and r.json()["executor"] == "worker"
    assert ctx.repo.next_queued_run()["params"] == {"force_publish": True}
    assert should_trigger(ctx, 0.0, None) is None             # nothing new to learn from


def test_worker_policy_threshold_and_min_gap(client, ctx):
    headers = register(client)
    _ingest(client, headers, [device_event(1)] + [interaction_event(10 + i, 20.0, "NOP", "nop") for i in range(3)])
    ctx.settings.evolution_min_new_logs = 3
    assert should_trigger(ctx, 10_000.0, None) == "threshold"
    assert should_trigger(ctx, 10_000.0, 10_000.0 - 60) is None        # a run started a minute ago
    assert should_trigger(ctx, 10_000.0, 10_000.0 - 601) == "threshold"
    ctx.settings.evolution_min_new_logs = 100
    assert should_trigger(ctx, 10_000.0, 10_000.0 - 601) is None       # under threshold, schedule not due
    assert should_trigger(ctx, 100_000.0, 0.0) == "schedule"
