"""INT16 model image <-> float weights.

Byte layout = gemma4/gemma4_slice_int16.py = what gemma4_engine.h maps:
  [64-byte header] [token table] [layer 0..N-1] [final norm]
  matrix  = float32 row scales[out] + int16 weights[out][in]  (scale = max|row| / 32767)
  norms   = float32[H] (Gemma RMSNorm uses (1 + w))
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .model_spec import (
    IMAGE_FLAG_TIED, IMAGE_FLAG_TOK_LISA, IMAGE_FORMAT_VERSION, IMAGE_HEADER_BYTES, IMAGE_HEADER_FMT, IMAGE_MAGIC,
    PSRAM_MODEL_BUDGET, SPEC, W25_SLOT_BYTES, ModelSpec,
)

Weights = dict[str, np.ndarray]
LAYER_MATRICES = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")


def weight_shapes(spec: ModelSpec = SPEC) -> dict[str, tuple[int, ...]]:
    H, I = spec.hidden, spec.intermediate
    shapes: dict[str, tuple[int, ...]] = {"embed": (spec.vocab, H), "final_norm": (H,)}
    for l in range(spec.layers):
        p = f"layers.{l}."
        shapes.update({p + "attn_norm": (H,), p + "wq": (H, H), p + "wk": (H, H), p + "wv": (H, H),
                       p + "wo": (H, H), p + "ffn_norm": (H,), p + "w_gate": (I, H), p + "w_up": (I, H),
                       p + "w_down": (H, I)})
    return shapes


def random_weights(spec: ModelSpec = SPEC, seed: int = 42, std: float = 0.02) -> Weights:
    """Same distribution as gemma4_slice_int16.py (norm weights 0 = unit gain)."""
    rng = np.random.default_rng(seed)
    out: Weights = {}
    for name, shape in weight_shapes(spec).items():
        out[name] = np.zeros(shape, np.float32) if len(shape) == 1 else (
            rng.standard_normal(shape, dtype=np.float32) * std)
    return out


def quantize_q16(m: np.ndarray) -> bytes:
    m = np.asarray(m, dtype=np.float32)
    scales = np.max(np.abs(m), axis=1) / 32767.0
    scales[scales == 0.0] = 1.0
    scales = scales.astype(np.float32)
    q = np.clip(np.round(m / scales[:, None]), -32767, 32767).astype(np.int16)
    return scales.astype("<f4").tobytes() + q.astype("<i2").tobytes()


def dequantize_q16(buf: bytes, off: int, out_dim: int, in_dim: int) -> tuple[np.ndarray, int]:
    scales = np.frombuffer(buf, "<f4", out_dim, off).astype(np.float32)
    off += out_dim * 4
    q = np.frombuffer(buf, "<i2", out_dim * in_dim, off).reshape(out_dim, in_dim).astype(np.float32)
    off += out_dim * in_dim * 2
    return q * scales[:, None], off


@dataclass(frozen=True)
class ImageInfo:
    model_version: int
    size: int
    payload_crc32: int
    sha256: str
    flags: int = IMAGE_FLAG_TIED

    @property
    def lisa_tokenizer(self) -> bool:
        return bool(self.flags & IMAGE_FLAG_TOK_LISA)


def encode_image(w: Weights, model_version: int, spec: ModelSpec = SPEC, *, lisa_tokenizer: bool = True) -> bytes:
    """lisa_tokenizer: the weights were trained with lisa_tokenizer.bin (flag in the header, so the
    firmware and the server pick the matching tokenizer); False only for legacy byte-token models."""
    shapes = weight_shapes(spec)
    for name, shape in shapes.items():
        if tuple(w[name].shape) != shape:
            raise ValueError(f"{name}: shape {w[name].shape}, expected {shape}")
    parts = [quantize_q16(w["embed"])]
    for l in range(spec.layers):
        p = f"layers.{l}."
        parts.append(np.asarray(w[p + "attn_norm"], "<f4").tobytes())
        parts += [quantize_q16(w[p + m]) for m in ("wq", "wk", "wv", "wo")]
        parts.append(np.asarray(w[p + "ffn_norm"], "<f4").tobytes())
        parts += [quantize_q16(w[p + m]) for m in ("w_gate", "w_up", "w_down")]
    parts.append(np.asarray(w["final_norm"], "<f4").tobytes())
    payload = b"".join(parts)
    if len(payload) != spec.payload_bytes:
        raise AssertionError(f"payload {len(payload)} != layout {spec.payload_bytes}")
    header = struct.pack(IMAGE_HEADER_FMT, IMAGE_MAGIC, IMAGE_FORMAT_VERSION, IMAGE_HEADER_BYTES,
                         spec.hidden, spec.intermediate, spec.heads, spec.layers, spec.vocab,
                         len(payload), zlib.crc32(payload),
                         IMAGE_FLAG_TIED | (IMAGE_FLAG_TOK_LISA if lisa_tokenizer else 0), model_version, 0, 0, 0, 0, 0)
    image = header + payload
    check_budget(image)
    return image


def read_header(image: bytes, spec: ModelSpec = SPEC) -> ImageInfo:
    if len(image) < IMAGE_HEADER_BYTES:
        raise ValueError("image shorter than its header")
    (magic, fmt, hbytes, H, I, heads, L, V, payload, crc, flags, model_version, *_r) = struct.unpack_from(
        IMAGE_HEADER_FMT, image)
    if magic != IMAGE_MAGIC or fmt != IMAGE_FORMAT_VERSION or hbytes != IMAGE_HEADER_BYTES or not flags & IMAGE_FLAG_TIED:
        raise ValueError("not a W25Q model image (bad magic/version/flags)")
    if (H, I, heads, L, V) != (spec.hidden, spec.intermediate, spec.heads, spec.layers, spec.vocab):
        raise ValueError(f"image shape H={H} I={I} heads={heads} L={L} V={V} doesn't match the model spec")
    if payload != spec.payload_bytes or len(image) != hbytes + payload:
        raise ValueError("image size doesn't match its header")
    if zlib.crc32(image[hbytes:]) != crc:
        raise ValueError("payload CRC mismatch")
    return ImageInfo(model_version, len(image), crc, hashlib.sha256(image).hexdigest(), flags)


def decode_image(image: bytes, spec: ModelSpec = SPEC) -> Weights:
    read_header(image, spec)
    H, I = spec.hidden, spec.intermediate
    off = IMAGE_HEADER_BYTES
    w: Weights = {}
    w["embed"], off = dequantize_q16(image, off, spec.vocab, H)
    for l in range(spec.layers):
        p = f"layers.{l}."
        w[p + "attn_norm"] = np.frombuffer(image, "<f4", H, off).astype(np.float32)
        off += H * 4
        for m in ("wq", "wk", "wv", "wo"):
            w[p + m], off = dequantize_q16(image, off, H, H)
        w[p + "ffn_norm"] = np.frombuffer(image, "<f4", H, off).astype(np.float32)
        off += H * 4
        w[p + "w_gate"], off = dequantize_q16(image, off, I, H)
        w[p + "w_up"], off = dequantize_q16(image, off, I, H)
        w[p + "w_down"], off = dequantize_q16(image, off, H, I)
    w["final_norm"] = np.frombuffer(image, "<f4", H, off).astype(np.float32)
    return w


def set_model_version(image: bytes, model_version: int) -> bytes:
    """Rewrites only the header's model_version (the payload CRC is unaffected)."""
    read_header(image)
    off = struct.calcsize("<IHH8I")
    return image[:off] + struct.pack("<I", model_version) + image[off + 4:]


def check_budget(image: bytes) -> None:
    if len(image) > W25_SLOT_BYTES:
        raise ValueError(f"image is {len(image)} bytes; a W25Q slot holds {W25_SLOT_BYTES}")
    if len(image) - IMAGE_HEADER_BYTES > PSRAM_MODEL_BUDGET:
        raise ValueError("payload exceeds the device's PSRAM budget")
