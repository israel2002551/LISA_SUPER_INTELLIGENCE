from __future__ import annotations

import hashlib

import numpy as np
import pytest

from app.evolution import image_codec
from app.evolution.action_schema import arg_matches, validate_action
from app.evolution.delta import HEADER_BYTES, apply_delta, build_delta, changed_blocks
from app.evolution.model_spec import SPEC
from app.evolution.prompt_format import build_prompt, canonicalize_output, prompt_tokens, training_sequence
from app.evolution.signing import ManifestSigner, manifest_message
from app.evolution.teacher import DeviceContext, RuleTeacher
from app.evolution.tokenizer import BpeTokenizer

from .conftest import THERMO_MANIFEST, TOKENIZER


@pytest.fixture(scope="module")
def tok() -> BpeTokenizer:
    return BpeTokenizer.from_file(TOKENIZER)


# ---------------- tokenizer ----------------
def test_tokenizer_round_trips_ascii_within_the_truncated_vocab(tok):
    for text in ["hello world", '<d>thermo-01:thermostat<e>{"ev":"temp","v":31.2}<a>', "NOP", '{"cmd":"set_fan"}']:
        ids = tok.encode(text)
        assert ids and all(0 <= i < SPEC.vocab for i in ids)
        assert tok.decode(ids) == text.encode()


def test_tokenizer_byte_fallback_and_firmware_truncation(tok):
    ids = tok.encode("é")                       # 2 UTF-8 bytes -> byte tokens <0xC3><0xA9>
    assert tok.decode(ids) == "é".encode()
    assert tok.encode("abcdef", max_tokens=3) == tok.encode("abc")   # firmware truncates BYTES first


def test_prompt_contract(tok):
    long_event = "x" * 300 + "\r\n"
    p = build_prompt("thermo-01", "thermostat", long_event)
    assert p == b"<d>thermo-01:thermostat<e>" + b"x" * 120 + b"<a>"
    assert build_prompt("d", "t", "a\nb") == b"<d>d:t<e>a b<a>"
    toks = prompt_tokens(tok, p)
    assert toks[0] == SPEC.bos_id and len(toks) <= 96
    seq, n_prompt = training_sequence(tok, p, "NOP")
    assert seq[-1] == SPEC.eos_id and seq[:n_prompt] == toks


def test_actions_fit_the_device_token_budget(tok):
    """Regression: with a 32-token budget every set_fan action was cut off mid-JSON."""
    from app.evolution.model_spec import ACTION_MAX_TOKENS, PROMPT_MAX_TOKENS
    from app.evolution.prompt_format import target_fits

    action = '{"cmd":"set_fan","args":{"level":2}}'
    assert target_fits(tok, action) and len(tok.encode(action)) < ACTION_MAX_TOKENS
    assert not target_fits(tok, '{"cmd":"x","args":{"note":"' + "z" * 80 + '"}}')
    assert PROMPT_MAX_TOKENS + ACTION_MAX_TOKENS <= SPEC.max_context


def test_canonical_actions():
    assert canonicalize_output(' {"args":{"b":1,"a":2},"cmd":"x"}') == '{"cmd":"x","args":{"a":2,"b":1}}'
    assert canonicalize_output("NOP trailing") == "NOP"
    assert canonicalize_output("garbage") is None


# ---------------- action validation (twin of edge/action_schema.h) ----------------
@pytest.mark.parametrize("spec,value,ok", [
    ("int:0..3", 2, True), ("int:0..3", 4, False), ("int:0..3", 2.5, False), ("int:0..3", True, False),
    ("int:-10..10", -5, True), ("float:-40..125", 21.5, True), ("float:-40..125", 200, False),
    ("bool", False, True), ("bool", 0, False), ("str:4", "abcd", True), ("str:4", "abcde", False),
    ("enum:auto|off", "auto", True), ("enum:auto|off", "on", False), ("bogus", 1, False),
])
def test_arg_specs(spec, value, ok):
    assert arg_matches(spec, value) is ok


def test_validate_action():
    assert validate_action(THERMO_MANIFEST, '{"cmd":"set_fan","args":{"level":2}}')[:2] == (
        "ok", '{"cmd":"set_fan","args":{"level":2}}')
    assert validate_action(THERMO_MANIFEST, "NOP")[0] == "nop"
    assert validate_action(THERMO_MANIFEST, '{"cmd":"set_fan","args":{"level":9}}')[0] == "invalid"
    assert validate_action(THERMO_MANIFEST, '{"cmd":"set_fan","args":{}}')[2] == "missing arg 'level'"
    assert validate_action(THERMO_MANIFEST, '{"cmd":"reboot"}')[0] == "invalid"
    assert validate_action(THERMO_MANIFEST, '{"cmd":"beep"}')[0] == "ok"


def test_rule_teacher():
    t = RuleTeacher()
    dev = DeviceContext.from_row({"device_key": "thermo-01", "identified": True, "capability_manifest": THERMO_MANIFEST})
    assert t.label(dev, '{"ev":"temp","v":31}').target == '{"cmd":"set_fan","args":{"level":2}}'
    assert t.label(dev, '{"ev":"temp","v":10}').target == '{"cmd":"set_fan","args":{"level":0}}'
    assert t.label(dev, '{"ev":"temp","v":22}').target == "NOP"
    assert t.label(dev, "not json") is None
    raw = DeviceContext(key="raw:uart1", identified=False)
    assert t.label(raw, "$GPGGA,...").target == "NOP"     # never act on unidentified devices
    no_rules = DeviceContext(key="x", identified=True)
    assert t.label(no_rules, '{"ev":"temp","v":31}') is None


# ---------------- image codec ----------------
def test_image_round_trip_and_header(base_image):
    info = image_codec.read_header(base_image)
    assert info.size == len(base_image) == 64 + SPEC.payload_bytes
    w = image_codec.decode_image(base_image)
    assert set(w) == set(image_codec.weight_shapes())
    v7 = image_codec.set_model_version(base_image, 7)
    assert image_codec.read_header(v7).model_version == 7
    assert v7[64:] == base_image[64:] and info.payload_crc32 == image_codec.read_header(v7).payload_crc32
    w2 = image_codec.decode_image(image_codec.encode_image(w, 3))
    for k in w:   # re-quantizing dequantized weights is stable to within one INT16 step
        assert np.max(np.abs(w2[k] - w[k])) <= np.max(np.abs(w[k])) / 32767 * 1.01 + 1e-12


def test_image_rejects_corruption(base_image):
    bad = bytearray(base_image)
    bad[1000] ^= 0xFF
    with pytest.raises(ValueError, match="CRC"):
        image_codec.read_header(bytes(bad))


# ---------------- delta ----------------
def test_delta_round_trip_and_size(base_image):
    w = image_codec.decode_image(base_image)
    w["final_norm"] = w["final_norm"] + 0.01          # only the last block changes
    target = image_codec.encode_image(w, 2)
    delta = build_delta(base_image, target)
    assert apply_delta(base_image, delta) == target
    assert changed_blocks(delta) == 2                 # header block (version/CRC) + final-norm block
    assert len(delta) < HEADER_BYTES + 200 + 2 * 4096
    with pytest.raises(ValueError, match="different base"):
        apply_delta(target, delta)


# ---------------- signing (verified on-device by edge/ota_verify.h) ----------------
def test_manifest_signatures():
    s = ManifestSigner.generate()
    sha = hashlib.sha256(b"x").hexdigest()
    sig = s.sign(3, 5549120, sha)
    assert s.verify(3, 5549120, sha, sig)
    assert not s.verify(4, 5549120, sha, sig)
    assert manifest_message(3, 5549120, sha) == f"W25OTA1|3|5549120|{sha}".encode()
    assert s.public_pem_c_string().startswith('"-----BEGIN PUBLIC KEY-----\\n"')


def test_blob_upload_retries_dropped_connections(monkeypatch):
    import httpx

    from app.db import supabase as sb

    monkeypatch.setattr(sb.time, "sleep", lambda _s: None)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("x-upsert"))
        if len(calls) < 3:
            raise httpx.ReadError("connection reset", request=request)
        return httpx.Response(200, json={"Key": "models/v2.w25img"})

    store = sb.SupabaseBlobStore("https://example.supabase.co", "k")
    store._http = httpx.Client(transport=httpx.MockTransport(handler))
    store.put("models", "v2.w25img", b"\0" * 1024)
    assert calls == ["true"] * 3          # two resets, then success; every attempt upserts

    calls.clear()

    def always_down(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("down", request=request)

    store._http = httpx.Client(transport=httpx.MockTransport(always_down))
    try:
        store.put("models", "v2.w25img", b"x")
        raise AssertionError("expected ConnectError")
    except httpx.ConnectError:
        pass
    assert len(calls) == len(sb.RETRY_DELAYS_S) + 1


def test_synth_covers_the_whole_range_balanced_and_stable():
    """Boundary-only synthesis taught a real board 'NOP for everything'; the quota must be
    filled across the range so every rule region is well represented."""
    import collections
    import json

    from app.evolution.synth import synthesize

    dev = DeviceContext(key="sim-thermo", type="thermostat", identified=True, data_format="json_lines",
                        manifest=THERMO_MANIFEST, rules=list(THERMO_MANIFEST["rules"]))
    rows = synthesize([dev], 400)
    assert len(rows) == 400
    assert [r["dedupe_key"] for r in rows] == [r["dedupe_key"] for r in synthesize([dev], 400)]   # no growth per run
    temps = [json.loads(r["context_json"]["event"])["v"] for r in rows if '"temp"' in r["context_json"]["event"]]
    assert min(temps) < 15 - 10 and max(temps) > 30 + 10
    counts = collections.Counter(r["target_response"] for r in rows)
    assert len(counts) >= 3 and min(counts.values()) >= 0.2 * len(rows), counts
    assert sum(r["split"] == "val" for r in rows) >= 20
