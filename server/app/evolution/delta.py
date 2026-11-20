"""Block deltas for incremental W25Q updates (applied by edge/net_sync.h).

Format (little-endian):
  "W25D" u32 format=1  u32 block_size  u32 target_size
  base_sha256[32]  target_sha256[32]  u32 n_blocks
  bitmap[ceil(n_blocks/8)]   bit i (LSB first) = block i changed
  data of each changed block, in order (the last block may be short)

The device streams it: unchanged blocks are copied from the active slot, so it
needs no RAM beyond one block. Most useful when training freezes the token
table (2 MiB of the 5.3 MiB image stays identical).
"""

from __future__ import annotations

import hashlib
import struct

MAGIC = b"W25D"
FORMAT = 1
BLOCK = 4096
HEADER_BYTES = 84


def build_delta(base: bytes, target: bytes, block: int = BLOCK) -> bytes:
    n = (len(target) + block - 1) // block
    bitmap = bytearray((n + 7) // 8)
    chunks: list[bytes] = []
    for i in range(n):
        t = target[i * block:(i + 1) * block]
        if base[i * block:(i + 1) * block] != t:   # also true past the end of a shorter base
            bitmap[i // 8] |= 1 << (i % 8)
            chunks.append(t)
    header = (MAGIC + struct.pack("<III", FORMAT, block, len(target)) + hashlib.sha256(base).digest()
              + hashlib.sha256(target).digest() + struct.pack("<I", n))
    assert len(header) == HEADER_BYTES
    return header + bytes(bitmap) + b"".join(chunks)


def apply_delta(base: bytes, delta: bytes) -> bytes:
    """Reference implementation of the device's apply_delta()."""
    if delta[:4] != MAGIC:
        raise ValueError("not a W25D delta")
    fmt, block, size = struct.unpack_from("<III", delta, 4)
    base_sha, target_sha = delta[16:48], delta[48:80]
    (n,) = struct.unpack_from("<I", delta, 80)
    if fmt != FORMAT or n != (size + block - 1) // block:
        raise ValueError("bad delta header")
    if hashlib.sha256(base).digest() != base_sha:
        raise ValueError("delta was built against a different base image")
    bitmap = delta[HEADER_BYTES:HEADER_BYTES + (n + 7) // 8]
    off = HEADER_BYTES + len(bitmap)
    out = bytearray()
    for i in range(n):
        length = min(block, size - i * block)
        if bitmap[i // 8] & (1 << (i % 8)):
            out += delta[off:off + length]
            off += length
        else:
            out += base[i * block:i * block + length]
    if hashlib.sha256(out).digest() != target_sha:
        raise ValueError("patched image fails its SHA-256")
    return bytes(out)


def changed_blocks(delta: bytes) -> int:
    (n,) = struct.unpack_from("<I", delta, 80)
    bitmap = delta[HEADER_BYTES:HEADER_BYTES + (n + 7) // 8]
    return sum(bin(b).count("1") for b in bitmap)
