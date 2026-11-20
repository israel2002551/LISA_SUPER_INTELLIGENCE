from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.memory import InMemoryRepository, LocalBlobStore
from app.evolution import image_codec
from app.evolution.pipeline import EvolutionContext
from app.evolution.signing import ManifestSigner
from app.evolution.teacher import RuleTeacher
from app.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER = REPO_ROOT / "tokenizer.bin"
ADMIN = {"Authorization": "Bearer dev-admin-token"}

THERMO_MANIFEST = {
    "id": "thermo-01", "type": "thermostat", "fw": "1.0", "emits": ["temp", "hum"],
    "accepts": [{"cmd": "set_fan", "args": {"level": "int:0..3"}}, {"cmd": "beep", "args": {}}],
    "needs": ["alerts"], "acks": True, "bus": {"max_baud": 2000000, "links": ["uart"]},
    "rules": [
        {"when": {"ev": "temp", "gt": 30}, "do": {"cmd": "set_fan", "args": {"level": 2}}},
        {"when": {"ev": "temp", "lt": 15}, "do": {"cmd": "set_fan", "args": {"level": 0}}},
    ],
}


@pytest.fixture(scope="session")
def signer() -> ManifestSigner:
    return ManifestSigner.generate()


@pytest.fixture(scope="session")
def base_image() -> bytes:
    """A valid model image (random weights, like gemma4_slice_int16.py), version 0."""
    return image_codec.encode_image(image_codec.random_weights(seed=42), model_version=0)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path, tokenizer_path=TOKENIZER, trainer="mock",
                    min_train_examples=1, public_base_url="http://testserver", run_evolution_inline=False)


@pytest.fixture
def ctx(settings, signer, tmp_path) -> EvolutionContext:
    blobs = LocalBlobStore(tmp_path / "blobs", settings.public_base_url, b"test-secret")
    return EvolutionContext(settings=settings, repo=InMemoryRepository(), blobs=blobs, signer=signer,
                            teacher=RuleTeacher())


@pytest.fixture
def client(ctx) -> TestClient:
    return TestClient(create_app(ctx))


def register(client: TestClient, node_id: str = "edge-01") -> dict[str, str]:
    r = client.post("/api/v1/admin/nodes", json={"node_id": node_id}, headers=ADMIN)
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['api_key']}", "X-Node-Id": node_id}


def device_event(seq: int, manifest: dict = THERMO_MANIFEST, baud: int = 2000000) -> dict:
    return {"seq": seq, "type": 2, "data": {
        "kind": "device", "dev": manifest["id"], "port": 1, "identified": True, "baud": baud,
        "link": "uart_fast", "format": "json_lines", "type": manifest["type"], "manifest": manifest}}


def interaction_event(seq: int, temp: float, output: str, status: str, act_id: int | None = None) -> dict:
    from app.evolution.prompt_format import build_prompt

    event = json.dumps({"ev": "temp", "v": temp}, separators=(",", ":"))
    return {"seq": seq, "type": 1, "data": {
        "kind": "interaction", "dev": "thermo-01", "act_id": act_id or seq, "input": event,
        "prompt": build_prompt("thermo-01", "thermostat", event).decode(), "output": output,
        "status": status, "lat": 80, "mv": 1}}


def status_block(sha: str = "", version: int = 0) -> dict:
    return {"model_version": version, "model_loaded": True,
            "partitions": {"active_slot": 0, "slots": [{"slot": "A", "state": "good", "version": version, "sha256": sha},
                                                      {"slot": "B", "state": "empty", "version": 0, "sha256": ""}]},
            "w25q": {"jedec_id": "ef4018", "size": 16777216}}
