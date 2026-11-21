"""The server and the firmware must agree byte-for-byte. These tests read the C
headers and fail if anyone changes one side without the other."""

from __future__ import annotations

import importlib.util
import re
import struct
import zlib

from app.evolution import image_codec
from app.evolution.model_spec import (
    ACTION_MAX_TOKENS, EVENT_MAX_CHARS, IMAGE_HEADER_BYTES, IMAGE_MAGIC, PROMPT_MAX_TOKENS, SPEC, W25_SLOT_BYTES,
)

from .conftest import REPO_ROOT


def defines(path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"^\s*#define\s+(\w+)\s+([^\s/]+)", text, re.M)}


def num(v: str) -> float:
    return float(int(v.rstrip("uU"), 0)) if re.match(r"^(0x[0-9a-fA-F]+|\d+)[uU]?$", v) else float(v.rstrip("fF"))


def test_model_shape_matches_firmware():
    d = defines(REPO_ROOT / "gemma4" / "gemma4_w25q128_config.h")
    assert num(d["GEMMA_HIDDEN_DIM"]) == SPEC.hidden
    assert num(d["GEMMA_INTERMEDIATE_DIM"]) == SPEC.intermediate
    assert num(d["GEMMA_NUM_HEADS"]) == SPEC.heads
    assert num(d["GEMMA_HEAD_DIM"]) == SPEC.head_dim
    assert num(d["GEMMA_VOCAB_SIZE"]) == SPEC.vocab
    assert num(d["TOTAL_TRANSFORMER_LAYERS"]) == SPEC.layers
    assert num(d["GEMMA_MAX_CONTEXT_LENGTH"]) == SPEC.max_context
    assert num(d["GEMMA_BOS_TOKEN_ID"]) == SPEC.bos_id
    assert num(d["GEMMA_EOS_TOKEN_ID"]) == SPEC.eos_id
    assert num(d["GEMMA_QUERY_SCALE"]) == SPEC.query_scale
    assert num(d["GEMMA_EMBED_SCALE"]) == SPEC.embed_scale
    assert num(d["W25_MODEL_MAGIC"]) == IMAGE_MAGIC
    assert num(d["W25_MODEL_HEADER_BYTES"]) == IMAGE_HEADER_BYTES
    from app.evolution.model_spec import IMAGE_FLAG_TIED, IMAGE_FLAG_TOK_LISA
    assert num(d["W25_MODEL_FLAG_TIED"]) == IMAGE_FLAG_TIED
    assert num(d["W25_MODEL_FLAG_TOK_LISA"]) == IMAGE_FLAG_TOK_LISA


def test_prompt_contract_and_slot_size_match_firmware():
    e = defines(REPO_ROOT / "edge" / "edge_config.h")
    assert num(e["EDGE_EVENT_MAX_CHARS"]) == EVENT_MAX_CHARS
    assert num(e["EDGE_PROMPT_MAX_TOKENS"]) == PROMPT_MAX_TOKENS
    assert num(e["EDGE_ACTION_MAX_TOKENS"]) == ACTION_MAX_TOKENS
    from app.evolution.model_spec import CHAT_MAX_CHARS
    assert num(e["EDGE_CHAT_MAX_CHARS"]) == CHAT_MAX_CHARS
    schema = (REPO_ROOT / "edge" / "action_schema.h").read_text(encoding="utf-8")
    assert '"<u>%s<l>"' in schema                      # build_chat_prompt's tags
    layout = defines(REPO_ROOT / "edge" / "w25q_layout.h")
    assert num(layout["W25_SLOT_BYTES"]) == W25_SLOT_BYTES


def test_codec_writes_the_same_bytes_as_the_slicer():
    """Same float weights (both draw from the same RNG in the same order) -> identical images."""
    spec = importlib.util.spec_from_file_location("slicer", REPO_ROOT / "gemma4" / "gemma4_slice_int16.py")
    slicer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slicer)
    cfg = slicer.read_config(slicer.CONFIG_H)
    payload = slicer.build_payload(cfg, seed=7)
    header = struct.pack(slicer.HEADER_FMT, cfg["magic"], cfg["version"], cfg["header_bytes"], cfg["hidden"],
                         cfg["intermediate"], cfg["heads"], cfg["layers"], cfg["vocab"], len(payload),
                         zlib.crc32(payload), cfg["flag_tied"], 0, 0, 0, 0, 0, 0)
    # The slicer writes untrained images, which carry no tokenizer flag.
    assert image_codec.encode_image(image_codec.random_weights(seed=7), model_version=0,
                                    lisa_tokenizer=False) == header + payload
