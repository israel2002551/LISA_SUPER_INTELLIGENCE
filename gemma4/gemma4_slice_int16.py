"""
gemma4_slice_int16.py - builds the INT16 model image for the external W25Q128
==============================================================================
Writes one binary that w25q128_upload.py puts on the W25Q128, starting at
address 0. (The chip is on the ESP32-S3's SPI2 pins, so esptool can't reach it.)
At boot the firmware copies it into PSRAM, so it must fit there (about 5.3 MB
with the default shape).

The model shape is read from gemma4_w25q128_config.h, so the image always
matches the firmware. The firmware also checks the header at boot.

Image layout (little-endian):
  [64-byte header]   magic "G4W1", version, shape, payload size, payload CRC-32
  token table        float32 scales[V] + int16 weights[V][H]
                     (input embeddings AND output classifier: tied, as in Gemma)
  N layers           attn_norm f32[H], wq, wk, wv, wo, ffn_norm f32[H],
                     w_gate, w_up, w_down
                     (each matrix = float32 row scales[out] + int16 weights[out][in])
  final norm         float32[H]

NOTE: the weights are random. No pretrained checkpoint exists for this shape,
so the firmware produces gibberish until real weights are trained or converted
into this layout.
"""

import argparse
import os
import re
import struct
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_H = os.path.join(HERE, "gemma4_w25q128_config.h")
W25Q128_BYTES = 16 * 1024 * 1024
HEADER_FMT = "<IHH8I6I"   # must match struct W25ModelHeader


def read_config(path):
    """Pull the model shape and image constants out of the firmware header."""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    def define(name):
        m = re.search(rf"^\s*#define\s+{name}\s+(0x[0-9A-Fa-f]+|\d+)u?\b", text, re.M)
        if not m:
            sys.exit(f"error: #define {name} not found in {path}")
        return int(m.group(1), 0)

    return {
        "hidden": define("GEMMA_HIDDEN_DIM"),
        "intermediate": define("GEMMA_INTERMEDIATE_DIM"),
        "heads": define("GEMMA_NUM_HEADS"),
        "layers": define("TOTAL_TRANSFORMER_LAYERS"),
        "vocab": define("GEMMA_VOCAB_SIZE"),
        "magic": define("W25_MODEL_MAGIC"),
        "version": define("W25_MODEL_VERSION"),
        "header_bytes": define("W25_MODEL_HEADER_BYTES"),
        "flag_tied": define("W25_MODEL_FLAG_TIED"),
    }


def quantize_q16(arr):
    """Per-row INT16: float32 scales[out] followed by int16 weights[out][in]."""
    scales = np.max(np.abs(arr), axis=1) / 32767.0
    scales[scales == 0.0] = 1.0
    scales = scales.astype(np.float32)
    q16 = np.clip(np.round(arr / scales[:, None]), -32767, 32767).astype(np.int16)
    return scales.tobytes() + q16.tobytes()


def pack_layer(attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down):
    return b"".join([
        attn_norm.astype(np.float32).tobytes(),
        quantize_q16(wq), quantize_q16(wk), quantize_q16(wv), quantize_q16(wo),
        ffn_norm.astype(np.float32).tobytes(),
        quantize_q16(w_gate), quantize_q16(w_up), quantize_q16(w_down),
    ])


def expected_payload_bytes(cfg):
    """Mirror of GEMMA_MODEL_PAYLOAD_BYTES in gemma4_w25q128_config.h."""
    H, I, V, L = cfg["hidden"], cfg["intermediate"], cfg["vocab"], cfg["layers"]
    q = lambda out_dim, in_dim: out_dim * 4 + out_dim * in_dim * 2
    layer = 2 * H * 4 + 4 * q(H, H) + 2 * q(I, H) + q(H, I)
    return q(V, H) + L * layer + H * 4


def build_payload(cfg, seed):
    H, I, V, L = cfg["hidden"], cfg["intermediate"], cfg["vocab"], cfg["layers"]
    rng = np.random.default_rng(seed)

    def weights(rows, cols):
        return rng.standard_normal((rows, cols), dtype=np.float32) * 0.02

    # Gemma RMSNorm multiplies by (1 + w), so w = 0 means unit gain.
    unit_norm = np.zeros(H, dtype=np.float32)

    print(f" -> Token table (tied embeddings/classifier) [{V} x {H}]")
    parts = [quantize_q16(weights(V, H))]
    for l in range(L):
        print(f" -> Layer {l}/{L - 1}")
        parts.append(pack_layer(
            unit_norm, weights(H, H), weights(H, H), weights(H, H), weights(H, H),
            unit_norm, weights(I, H), weights(I, H), weights(H, I)))
    print(" -> Final norm")
    parts.append(unit_norm.tobytes())
    return b"".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Build the Gemma 4 INT16 image for an external W25Q128")
    parser.add_argument("--output", default="cluster_binaries/gemma4_w25q128_int16.bin", help="output image path")
    parser.add_argument("--config", default=CONFIG_H, help="firmware config header to read the shape from")
    parser.add_argument("--seed", type=int, default=42, help="seed for the random weights")
    args = parser.parse_args()

    cfg = read_config(args.config)
    if cfg["hidden"] % cfg["heads"]:
        sys.exit("error: hidden dim must be divisible by the number of heads")

    print(f"Building synthetic Gemma 4 INT16 image: H={cfg['hidden']} FFN={cfg['intermediate']} "
          f"heads={cfg['heads']} layers={cfg['layers']} vocab={cfg['vocab']} (tied embeddings)")
    payload = build_payload(cfg, args.seed)
    if len(payload) != expected_payload_bytes(cfg):
        sys.exit(f"error: payload is {len(payload)} bytes, layout says {expected_payload_bytes(cfg)}")

    header = struct.pack(
        HEADER_FMT, cfg["magic"], cfg["version"], cfg["header_bytes"],
        cfg["hidden"], cfg["intermediate"], cfg["heads"], cfg["layers"], cfg["vocab"],
        len(payload), zlib.crc32(payload), cfg["flag_tied"], *([0] * 6))
    if len(header) != cfg["header_bytes"]:
        sys.exit(f"error: header is {len(header)} bytes, expected {cfg['header_bytes']}")

    image = header + payload
    if len(image) > W25Q128_BYTES:
        sys.exit(f"error: image is {len(image)} bytes, larger than the 16 MB W25Q128")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(image)

    print(f"\n[Success] {args.output}: {len(image):,} bytes ({len(image) / 2**20:.2f} MiB), "
          f"payload CRC-32 {zlib.crc32(payload):08X}")
    print(f"Next: python gemma4/w25q128_upload.py <port> {args.output}")


if __name__ == "__main__":
    main()
