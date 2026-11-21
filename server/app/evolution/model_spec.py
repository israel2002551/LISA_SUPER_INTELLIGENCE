"""Model shape and image constants.

MUST match gemma4/gemma4_w25q128_config.h (tests/test_image_codec.py parses the
header and fails if they drift) and edge/w25q_layout.h (slot size).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    hidden: int = 256
    intermediate: int = 768
    heads: int = 4
    head_dim: int = 64
    vocab: int = 4096
    layers: int = 2
    max_context: int = 128
    bos_id: int = 1
    eos_id: int = 2
    byte_token_offset: int = 3
    query_scale: float = 0.125        # 1 / sqrt(head_dim)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    @property
    def embed_scale(self) -> float:   # sqrt(hidden), GEMMA_EMBED_SCALE
        return math.sqrt(self.hidden)

    # ---- image layout ----
    @staticmethod
    def q16_bytes(out_dim: int, in_dim: int) -> int:
        return out_dim * 4 + out_dim * in_dim * 2

    @property
    def table_bytes(self) -> int:
        return self.q16_bytes(self.vocab, self.hidden)

    @property
    def layer_bytes(self) -> int:
        H, I = self.hidden, self.intermediate
        return 2 * H * 4 + 4 * self.q16_bytes(H, H) + 2 * self.q16_bytes(I, H) + self.q16_bytes(H, I)

    @property
    def payload_bytes(self) -> int:
        return self.table_bytes + self.layers * self.layer_bytes + self.hidden * 4


SPEC = ModelSpec()

# Image header (struct W25ModelHeader)
IMAGE_MAGIC = 0x31573447          # "G4W1"
IMAGE_FORMAT_VERSION = 1
IMAGE_HEADER_BYTES = 64
IMAGE_FLAG_TIED = 0x1
IMAGE_FLAG_TOK_LISA = 0x2         # trained with lisa_tokenizer.bin; unset = legacy byte tokens (tokenizer.bin)
IMAGE_HEADER_FMT = "<IHH8I6I"     # magic, version, header_bytes, 8 x u32, model_version + 5 reserved

# edge/w25q_layout.h
W25_SLOT_BYTES = 0x600000
PSRAM_MODEL_BUDGET = 6 * 1024 * 1024

# Prompt contract (edge/edge_config.h)
EVENT_MAX_CHARS = 120
PROMPT_MAX_TOKENS = 80            # including BOS
ACTION_MAX_TOKENS = 48            # output budget (actions and chat replies); an action is ~21 tokens

# Chat contract (edge/action_schema.h edge_build_chat_prompt): "<u>" + text + "<l>" -> reply, EOS
CHAT_USER_TAG = b"<u>"
CHAT_LISA_TAG = b"<l>"
CHAT_MAX_CHARS = 150              # bytes of user text kept

QUANTIZATION_FORMAT = "q16_rowscale_tied"
BASE_MODEL = "lisa-gemma4-int16-h256-l2-v4096"
