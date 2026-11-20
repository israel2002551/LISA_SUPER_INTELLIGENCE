#pragma once
#include <stdint.h>

// ==============================================================================
// Gemma 4 single-chip build: ESP32-S3 + external W25Q128 on SPI2
// ==============================================================================
// This build does not use gemma4_cluster_config.h (that one is for the 8-node
// ESP-NOW mesh). The model here is much smaller because the W25Q128 sits on SPI2
// and can't be memory-mapped: the whole model is copied into PSRAM at boot, so it
// has to fit in PSRAM (8 MB on an N16R8 module) next to the KV cache.
//
// gemma4_slice_int16.py reads the #defines below to build a matching image, and
// the firmware checks the image header against them at boot.
// ==============================================================================

// --- Model shape --------------------------------------------------------------
#define GEMMA_HIDDEN_DIM           256
#define GEMMA_INTERMEDIATE_DIM     768     // GeGLU FFN (3x hidden)
#define GEMMA_NUM_HEADS              4
#define GEMMA_HEAD_DIM              64     // GEMMA_HIDDEN_DIM / GEMMA_NUM_HEADS
#define GEMMA_VOCAB_SIZE          4096     // first 4096 entries of tokenizer.bin
#define TOTAL_TRANSFORMER_LAYERS     2
#define GEMMA_MAX_CONTEXT_LENGTH   128

// Special tokens follow the tokenizer actually used (llama2.c tokenizer.bin):
// 0 = <unk>, 1 = <s>, 2 = </s>, and byte tokens <0x00>..<0xFF> are 3..258.
#define GEMMA_BOS_TOKEN_ID           1
#define GEMMA_EOS_TOKEN_ID           2
#define GEMMA_BYTE_TOKEN_OFFSET      3

#define GEMMA_QUERY_SCALE          0.125f  // 1 / sqrt(GEMMA_HEAD_DIM)
#define GEMMA_EMBED_SCALE          16.0f   // sqrt(GEMMA_HIDDEN_DIM)

// --- Sampling -----------------------------------------------------------------
#define DEFAULT_TEMPERATURE        0.8f
#define DEFAULT_TOP_P              0.9f
#define REPETITION_PENALTY         1.15f
#define REPETITION_WINDOW          16
#define MAX_NEW_TOKENS             35

// Every prompt runs as [BOS] + system prompt + user text. The [BOS] + system part
// is computed once and restored from the prefix cache. Change it with /system.
#define GEMMA_DEFAULT_SYSTEM_PROMPT "You are LISA, a helpful AI."   // 27 tokens; the prefix cache holds 31

// --- Serial -------------------------------------------------------------------
// 921600 baud keeps w25q128_upload.py to roughly 1-2 minutes for a ~5 MB image.
#define SERIAL_BAUD                921600

// --- W25Q128 wiring (SPI2 / FSPI) ---------------------------------------------
// Defaults are the SPI2 IO_MUX pins. Override with -D flags in platformio.ini.
// Tie the W25Q128's WP# (pin 3) and HOLD# (pin 7) to 3.3 V, unless you wire them
// to GPIOs and switch W25_IO_MODE to SPI_FLASH_QIO.
#ifndef W25_PIN_CS
#define W25_PIN_CS     10
#endif
#ifndef W25_PIN_MOSI
#define W25_PIN_MOSI   11   // W25Q128 DI (IO0)
#endif
#ifndef W25_PIN_CLK
#define W25_PIN_CLK    12
#endif
#ifndef W25_PIN_MISO
#define W25_PIN_MISO   13   // W25Q128 DO (IO1)
#endif
#ifndef W25_PIN_WP
#define W25_PIN_WP     -1   // W25Q128 WP# (IO2), only needed for SPI_FLASH_QIO
#endif
#ifndef W25_PIN_HD
#define W25_PIN_HD     -1   // W25Q128 HOLD# (IO3), only needed for SPI_FLASH_QIO
#endif
#ifndef W25_IO_MODE
#define W25_IO_MODE    SPI_FLASH_DIO      // SPI_FLASH_FASTRD if DIO gives CRC errors
#endif
#ifndef W25_SPEED
#define W25_SPEED      ESP_FLASH_20MHZ    // ESP_FLASH_40MHZ works with short wires
#endif

// --- Model image on the W25Q128 (starts at address 0) ---------------------------
//   [64-byte header][token table][layer 0] .. [layer N-1][final norm]
// token table = float32 scales[V] + int16 weights[V][H]. It is both the input
// embedding table and the output classifier (tied, as in Gemma).
#define W25_MODEL_MAGIC          0x31573447u   // "G4W1"
#define W25_MODEL_VERSION        1
#define W25_MODEL_HEADER_BYTES   64
#define W25_MODEL_FLAG_TIED      0x1u

struct W25ModelHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t num_heads;
    uint32_t num_layers;
    uint32_t vocab_size;
    uint32_t payload_bytes;   // everything after the header
    uint32_t payload_crc32;   // zlib CRC-32 of the payload
    uint32_t flags;
    uint32_t model_version;   // model_versions.version from the evolution server (0 = unversioned)
    uint32_t reserved[5];
};

constexpr uint32_t gemma_q16_matrix_bytes(uint32_t out_dim, uint32_t in_dim) {
    return out_dim * 4u + out_dim * in_dim * 2u;   // float32 row scales + int16 weights
}
constexpr uint32_t GEMMA_TABLE_BYTES = gemma_q16_matrix_bytes(GEMMA_VOCAB_SIZE, GEMMA_HIDDEN_DIM);
constexpr uint32_t GEMMA_LAYER_BYTES =
    2u * GEMMA_HIDDEN_DIM * 4u +                                          // attn_norm, ffn_norm
    4u * gemma_q16_matrix_bytes(GEMMA_HIDDEN_DIM, GEMMA_HIDDEN_DIM) +       // wq, wk, wv, wo
    2u * gemma_q16_matrix_bytes(GEMMA_INTERMEDIATE_DIM, GEMMA_HIDDEN_DIM) + // w_gate, w_up
    gemma_q16_matrix_bytes(GEMMA_HIDDEN_DIM, GEMMA_INTERMEDIATE_DIM);       // w_down
constexpr uint32_t GEMMA_MODEL_PAYLOAD_BYTES =
    GEMMA_TABLE_BYTES + TOTAL_TRANSFORMER_LAYERS * GEMMA_LAYER_BYTES + GEMMA_HIDDEN_DIM * 4u;

static_assert(sizeof(W25ModelHeader) == W25_MODEL_HEADER_BYTES, "W25ModelHeader layout");
static_assert(GEMMA_HIDDEN_DIM == GEMMA_NUM_HEADS * GEMMA_HEAD_DIM, "hidden dim must split evenly into heads");
static_assert(GEMMA_VOCAB_SIZE <= 65535, "sampler stores token ids as uint16_t");
static_assert(W25_MODEL_HEADER_BYTES + GEMMA_MODEL_PAYLOAD_BYTES <= 16u * 1024u * 1024u,
              "model image must fit on a 16 MB W25Q128");
static_assert(GEMMA_MODEL_PAYLOAD_BYTES <= 6u * 1024u * 1024u,
              "model must leave room in 8 MB PSRAM for the KV cache, tokenizer and caches");
