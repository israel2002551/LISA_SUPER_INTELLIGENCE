"""Exact Python port of bpe_tokenizer.h (as used by the edge firmware).

Training must tokenize prompts exactly as the device does, or the model learns
from token sequences it will never see. This mirrors the firmware's quirks:
  * it works on bytes, and truncates the input to `max_tokens` BYTES before merging
  * single-byte lookups use the LAST matching vocab id, longer strings the FIRST
  * merges pick the highest score (strictly greater -> first pair wins ties)
  * merged strings of 64+ bytes are never considered
"""

from __future__ import annotations

import struct
from pathlib import Path


class BpeTokenizer:
    def __init__(self, data: bytes, vocab_size: int = 4096, bos_id: int = 1, eos_id: int = 2) -> None:
        self.bos_id, self.eos_id = bos_id, eos_id
        self.vocab: list[bytes] = []
        self.scores: list[float] = []
        off = 4   # int32 max_token_length
        for _ in range(vocab_size):
            score, length = struct.unpack_from("<fi", data, off)
            off += 8
            self.vocab.append(data[off:off + length])
            self.scores.append(score)
            off += length
        self._first: dict[bytes, int] = {}
        self._byte_last: dict[int, int] = {}
        for i, tok in enumerate(self.vocab):
            self._first.setdefault(tok, i)
            if len(tok) == 1:
                self._byte_last[tok[0]] = i

    @classmethod
    def from_file(cls, path: str | Path, vocab_size: int = 4096) -> "BpeTokenizer":
        return cls(Path(path).read_bytes(), vocab_size)

    def _find(self, s: bytes) -> int:
        if len(s) == 1 and s[0] in self._byte_last:
            return self._byte_last[s[0]]
        return self._first.get(s, -1)

    def encode(self, text: str | bytes, max_tokens: int | None = None) -> list[int]:
        data = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        if max_tokens is not None:
            data = data[:max_tokens]
        if not data:
            return []
        tokens: list[int] = []
        for b in data:
            tid = self._find(bytes([b]))
            if tid == -1:
                fb = self._find(f"<0x{b:02X}>".encode())
                tid = fb if fb != -1 else 0
            tokens.append(tid)

        while len(tokens) >= 2:
            best_score, best_idx, best_tok = -1e9, -1, -1
            for i in range(len(tokens) - 1):
                a, c = self.vocab[tokens[i]], self.vocab[tokens[i + 1]]
                if len(a) + len(c) >= 64:
                    continue
                tid = self._find(a + c)
                if tid != -1 and self.scores[tid] > best_score:
                    best_score, best_idx, best_tok = self.scores[tid], i, tid
            if best_idx == -1:
                break
            tokens[best_idx] = best_tok
            del tokens[best_idx + 1]
        return tokens

    def decode_token(self, tid: int) -> bytes:
        if tid >= len(self.vocab):
            return f"[{tid}]".encode()
        if tid in (self.bos_id, self.eos_id):
            return b""
        tok = self.vocab[tid]
        if len(tok) == 6 and tok.startswith(b"<0x"):
            return bytes([int(tok[3:5], 16)])
        return tok

    def decode(self, tokens: list[int]) -> bytes:
        return b"".join(self.decode_token(t) for t in tokens)
