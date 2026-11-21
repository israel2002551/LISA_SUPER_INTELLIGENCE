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


# ---------------- LISA tokenizer + chat ----------------

def test_lisa_tokenizer_has_characters_and_merges(tok):
    from .conftest import LEGACY_TOKENIZER

    assert tok.vocab[3] == b"<0x00>" and tok.vocab[258] == b"<0xFF>" and len(tok.vocab) == 4096
    assert all(bytes([c]) in tok._first for c in range(0x20, 0x7F))        # every printable character
    text = "Hello, I am LISA. How are you today?"
    ids = tok.encode(text)
    assert tok.decode(ids).decode() == text and len(ids) <= 16              # word pieces, not bytes
    legacy = BpeTokenizer.from_file(LEGACY_TOKENIZER)
    assert legacy.encode(text) == [3 + b for b in text.encode()]            # what models v1-v3 saw


def test_firmware_copy_of_the_tokenizer_matches_the_file():
    import re
    import zlib

    from .conftest import REPO_ROOT, TOKENIZER

    data = TOKENIZER.read_bytes()
    header = (REPO_ROOT / "edge" / "lisa_tokenizer_data.h").read_text(encoding="utf-8")
    assert int(re.search(r"kLisaTokenizerCrc32 = 0x([0-9a-f]+)u", header).group(1), 16) == zlib.crc32(data)
    body = header[header.index("kLisaTokenizer[] = {"):]
    assert bytes(int(h, 16) for h in re.findall(r"0x([0-9a-f]{2})\b", body)) == data


def test_image_flags_record_the_tokenizer():
    w = image_codec.random_weights(seed=3)
    assert image_codec.read_header(image_codec.encode_image(w, 1)).lisa_tokenizer
    assert not image_codec.read_header(image_codec.encode_image(w, 1, lisa_tokenizer=False)).lisa_tokenizer


def test_chat_prompt_contract():
    from app.evolution.prompt_format import build_chat_prompt, is_chat_prompt

    assert build_chat_prompt("  hello lisa\r\n") == b"<u>hello lisa<l>"
    assert build_chat_prompt("x" * 400) == b"<u>" + b"x" * 150 + b"<l>"
    assert is_chat_prompt(build_chat_prompt("hi")) and not is_chat_prompt(b"<d>a:b<e>{}<a>")


def test_chat_corpus_fits_the_device_and_splits_by_phrasing(tok):
    import collections

    from app.evolution.chat_corpus import INTENTS, chat_rows
    from app.evolution.model_spec import ACTION_MAX_TOKENS, PROMPT_MAX_TOKENS
    from app.evolution.prompt_format import prompt_tokens

    rows = chat_rows()
    assert [r["dedupe_key"] for r in rows] == [r["dedupe_key"] for r in chat_rows()]       # stable
    owner = {}
    for intent, phrasings, reply in INTENTS:
        assert len(tok.encode(reply.encode())) < ACTION_MAX_TOKENS, reply                  # + EOS fits
        for p in phrasings:
            assert owner.setdefault(p, intent) == intent, f"'{p}' is in {owner[p]} and {intent}"
    for r in rows:
        assert len(prompt_tokens(tok, r["prompt"].encode())) < PROMPT_MAX_TOKENS
    by_split = collections.Counter(r["split"] for r in rows)
    assert by_split["val"] >= 60 and by_split["train"] >= 8 * by_split["val"] // 2
    trained = {r["context_json"]["intent"] for r in rows if r["split"] == "train"}
    assert trained >= {i for i, _, _ in INTENTS}                                           # every intent learnable
    # all typed forms of one phrasing share a split, so val measures unseen phrasings
    splits = collections.defaultdict(set)
    for r in rows:
        if not r["prompt"].startswith("<u>") or r["context_json"]["intent"] not in owner.values():
            continue
        core = r["prompt"][3:-3].lower().removeprefix("lisa, ").rstrip("?!.").removesuffix(", lisa")
        splits[(r["context_json"]["intent"], core)].add(r["split"])
    assert all(len(s) == 1 for s in splits.values())


# ---------------- memory, chat curation, Claude teacher ----------------

def test_chat_memory_prompt_and_followups(tok):
    from app.evolution.chat_corpus import INTENTS, MORE_REPLIES, REPEAT_NO_CONTEXT, corpus_reply
    from app.evolution.model_spec import ACTION_MAX_TOKENS
    from app.evolution.prompt_format import build_chat_prompt, split_chat_prompt

    joke = "Why did the computer go to the doctor? Because it had a virus!"
    p = build_chat_prompt(" say that again ", joke)
    assert p == b"<l>" + joke.encode() + b"<u>say that again<l>"
    assert split_chat_prompt(p.decode()) == (joke, "say that again")
    assert build_chat_prompt("hi", "x" * 150) == b"<u>hi<l>"                  # too long: no memory
    assert build_chat_prompt("who are you?", joke) == b"<u>who are you?<l>"  # not a follow-up: no memory
    from app.evolution.prompt_format import is_followup, normalize_message
    assert normalize_message("  Lisa,  Tell   me MORE?! ") == "tell me more" and is_followup("WHY, lisa?")
    assert not is_followup("why is the sky blue")
    assert corpus_reply("Say that again, lisa!", joke) == joke
    assert corpus_reply("say that again") == REPEAT_NO_CONTEXT
    assert corpus_reply("why", joke) == MORE_REPLIES["joke"]
    for reply in MORE_REPLIES.values():
        assert len(tok.encode(reply.encode())) < ACTION_MAX_TOKENS, reply
    assert set(MORE_REPLIES) <= {i for i, _, _ in INTENTS}


class _FakeClaude:
    """Stands in for anthropic.Anthropic(): returns canned structured-output JSON."""

    def __init__(self, *payloads):
        from types import SimpleNamespace

        self.payloads, self.requests = list(payloads), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        import json
        from types import SimpleNamespace

        self.requests.append(kwargs)
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps(self.payloads.pop(0)))])


def test_claude_teacher_reviews_chats_and_writes_lessons():
    from app.evolution.teacher import ChatTurn, ClaudeTeacher

    fake = _FakeClaude(
        {"reviews": [{"index": 0, "verdict": "good", "better_reply": "Hello!", "confidence": 0.9},
                     {"index": 1, "verdict": "bad", "better_reply": "The capital of France is Paris.", "confidence": 0.95},
                     {"index": 2, "verdict": "bad", "better_reply": "x" * 300, "confidence": 0.9}]},   # unusable
        {"lessons": [{"topic": "Capital of France", "phrasings": ["What is the capital of France?", "capital of france",
                                                                  "where is paris"], "reply": "Paris is the capital."},
                     {"topic": "bad", "phrasings": ["only one"], "reply": "No."}]})
    t = ClaudeTeacher(fake, model="claude-opus-5-5")
    reviews = t.review_chats([ChatTurn(None, "hi", "Hello!"), ChatTurn(None, "capital of france?", "I like blue."),
                              ChatTurn(None, "x", "y")])
    assert reviews[0].verdict == "good" and reviews[1].better_reply == "The capital of France is Paris."
    assert reviews[2] is None
    req = fake.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["output_config"]["format"]["type"] == "json_schema"
    assert "Turn 1:" in req["messages"][0]["content"] and "LISA" in req["system"]
    lessons = t.lessons(5, ["greet"])
    assert [l.topic for l in lessons] == ["Capital of France"]
    assert lessons[0].phrasings[0] == "what is the capital of france" and t.calls == 2


def test_chat_logs_are_curated_by_you_then_the_corpus_then_the_teacher():
    from app.evolution.curator import curate_chats
    from app.evolution.teacher import ChatReview, RuleTeacher

    class Teacher(RuleTeacher):
        name = "claude-opus-5-5"

        def __init__(self):
            self.asked = []

        def review_chats(self, turns, batch=20):
            self.asked += [t.message for t in turns]
            return [ChatReview(index=0, verdict="bad", better_reply="The capital of France is Paris.", confidence=0.9)
                    for _ in turns]

    def log(i, msg, out, reward=None, prev=None):
        prompt = (f"<l>{prev}<u>{msg}<l>" if prev else f"<u>{msg}<l>")
        return {"id": i, "node_id": "n", "prompt": prompt, "model_output": out, "reward": reward,
                "payload": {"task": "chat"}, "model_version": 4}

    joke = "Why did the computer go to the doctor? Because it had a virus!"
    logs = [log(1, "hello", "Hello! I'm LISA. How can I help you today?"),                 # corpus agrees
            log(2, "how are you", "I like blue, like the little LED on my board."),           # corpus corrects
            log(3, "what is the capital of france", "I don't know that yet.", reward=None),  # teacher teaches
            log(4, "do you like pizza", "I like pizza!", reward=1.0),                         # you: /good
            log(5, "say that again", "Goodbye!", prev=joke)]                                  # memory: repeat
    stats = {"chat_real_log": 0, "chat_failure_recovery": 0, "chat_skipped": 0, "chat_review": []}
    t = Teacher()
    rows = {r["source_log_id"]: r for r in curate_chats(logs, t, stats)}
    assert t.asked == ["what is the capital of france", "do you like pizza"]   # only what the corpus can't answer
    assert rows[1]["source_type"] == "real_log" and rows[1]["teacher"] == "chat_corpus"
    assert rows[2]["target_response"].startswith("I'm doing well") and rows[2]["source_type"] == "failure_recovery"
    assert rows[3]["target_response"] == "The capital of France is Paris." and rows[3]["teacher"] == "claude-opus-5-5"
    assert rows[4]["target_response"] == "I like pizza!" and rows[4]["teacher"] == "user"
    assert rows[5]["target_response"] == joke and rows[5]["prompt"].startswith("<l>")
    assert stats["chat_real_log"] == 2 and stats["chat_failure_recovery"] == 3 and len(stats["chat_review"]) == 5
