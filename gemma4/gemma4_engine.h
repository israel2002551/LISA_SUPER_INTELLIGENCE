#pragma once
// ==============================================================================
// Gemma 4 INT16 inference engine (shared)
// ==============================================================================
// Used by:
//   gemma4/gemma4_w25q128_cluster_main.cpp   serial chat build
//   edge/edge_main.cpp                       self-evolving edge node
//
// Holds the model (a PSRAM copy of an image read from the W25Q128), the
// activation buffers, the transformer forward pass, the sampler and the
// tokenizer helpers. All state is global and NOT thread-safe: call the engine
// from one task only (the edge build runs it on the Core 1 inference task).
//
// The math here is mirrored exactly by server/app/evolution/model.py, which is
// how the evolution server trains weights that this code then runs.
// ==============================================================================

#include <Arduino.h>
#include <new>
#include "esp_partition.h"
#include "esp_spi_flash.h"      // spi_flash_mmap_handle_t, SPI_FLASH_MMAP_DATA (ESP-IDF 4.4)
#include "esp_heap_caps.h"
#include "esp_system.h"         // esp_random()

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "gemma4_w25q128_config.h"
#include "gemma4_int16_weights.h"
#include "gemma4_cache_subsystem.h"
#include "w25q128_flash.h"
#include "sha256_util.h"

// The tokenizer defaults to the llama2.c settings in cluster_config.h, which clash
// with this build. Override them: only the first GEMMA_VOCAB_SIZE entries of
// tokenizer.bin are loaded, so every token id the tokenizer produces is valid here.
#define BPE_VOCAB_SIZE   GEMMA_VOCAB_SIZE
#define BPE_BOS_TOKEN_ID GEMMA_BOS_TOKEN_ID
#define BPE_EOS_TOKEN_ID GEMMA_EOS_TOKEN_ID
#include "../bpe_tokenizer.h"

#ifndef GEMM_WORKER_PRIORITY
#define GEMM_WORKER_PRIORITY (configMAX_PRIORITIES - 1)
#endif

// Model weights (pointers into the PSRAM copy of the image)
struct Gemma4Int16Model {
    GemmaQuantizedTableQ16 token_table;   // tied: input embeddings and output classifier
    GemmaInt16LayerWeights layers[TOTAL_TRANSFORMER_LAYERS];
    const float* final_norm;
    uint32_t model_version;               // from the image header
    bool loaded;
};

static Gemma4Int16Model g_model;
static uint8_t* g_model_blob = nullptr;
static BpeTokenizer* g_tokenizer = nullptr;
static UnifiedCacheManager g_cache_mgr;

// Activation buffers (internal SRAM)
static float g_activation[GEMMA_HIDDEN_DIM];
static float g_norm_buf[GEMMA_HIDDEN_DIM];
static float g_q[GEMMA_HIDDEN_DIM];
static float g_k[GEMMA_HIDDEN_DIM];
static float g_v[GEMMA_HIDDEN_DIM];
static float g_attn_output[GEMMA_HIDDEN_DIM];
static float g_proj_output[GEMMA_HIDDEN_DIM];
static float g_h1_gate[GEMMA_INTERMEDIATE_DIM];
static float g_h3_up[GEMMA_INTERMEDIATE_DIM];
static float g_geglu_buf[GEMMA_INTERMEDIATE_DIM];
static float g_h2_down[GEMMA_HIDDEN_DIM];
static float g_logits[GEMMA_VOCAB_SIZE];

// Sampler scratch: candidate tokens sorted by probability
struct ProbIndex {
    float p;
    uint16_t id;
};
static ProbIndex g_candidates[GEMMA_VOCAB_SIZE];

// Repetition penalty window
static uint32_t g_recent_tokens[REPETITION_WINDOW];
static int g_recent_token_count = 0;

// ==============================================================================
// Startup
// ==============================================================================
// Starts the Core 0 half of the dual-core GEMM. Lower priority = Wi-Fi/UART tasks
// on Core 0 preempt it, so inference slows down instead of blocking I/O.
static void gemma_start_gemm_worker(UBaseType_t priority = GEMM_WORKER_PRIORITY) {
    if (g_core0_q16_handle) return;
    xTaskCreatePinnedToCore(core0_q16_gemm_worker, "Core0_Q16_GEMM", 4096,
                            xTaskGetCurrentTaskHandle(), priority, &g_core0_q16_handle, 0);
    while (!g_core0_q16_ready) delay(10);
}

// Loads tokenizer.bin from the ESP32-S3's own flash ('tokenizer' partition, memory-mapped).
static bool gemma_init_tokenizer() {
    if (g_tokenizer && g_tokenizer->initialized) return true;
    if (!g_tokenizer) {
        g_tokenizer = (BpeTokenizer*)heap_caps_malloc(sizeof(BpeTokenizer), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!g_tokenizer) g_tokenizer = (BpeTokenizer*)malloc(sizeof(BpeTokenizer));
        if (!g_tokenizer) return false;
        new (g_tokenizer) BpeTokenizer();
    }
    const esp_partition_t* part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_SPIFFS, "tokenizer");
    const void* ptr = nullptr;
    spi_flash_mmap_handle_t handle;
    return part && esp_partition_mmap(part, 0, part->size, SPI_FLASH_MMAP_DATA, &ptr, &handle) == ESP_OK &&
           g_tokenizer->init((const uint8_t*)ptr, part->size);
}

// ==============================================================================
// Model loading: W25Q128 -> PSRAM
// ==============================================================================
static void gemma_unload_model() {
    g_model.loaded = false;
    if (g_model_blob) {
        heap_caps_free(g_model_blob);
        g_model_blob = nullptr;
    }
}

static void gemma_map_layout(const uint8_t* base) {
    uint32_t offset = 0;
    g_model.token_table = GemmaQuantizedTableQ16::from_buffer(base, offset);
    for (int l = 0; l < TOTAL_TRANSFORMER_LAYERS; l++) {
        g_model.layers[l] = GemmaInt16LayerWeights::from_buffer(base, offset);
    }
    g_model.final_norm = (const float*)(base + offset);
}

// Reads and checks the header of the image at base_addr. Prints why it's rejected.
static bool gemma_read_image_header(uint32_t base_addr, W25ModelHeader& h, bool verbose = true) {
    if (!w25_read(base_addr, &h, sizeof(h))) {
        if (verbose) Serial.println("[Model] Could not read the image header from the W25Q128.");
        return false;
    }
    if (h.magic != W25_MODEL_MAGIC) {
        if (verbose) Serial.printf("[Model] No model image at 0x%06X.\n", (unsigned)base_addr);
        return false;
    }
    if (h.version != W25_MODEL_VERSION || h.header_bytes != W25_MODEL_HEADER_BYTES || !(h.flags & W25_MODEL_FLAG_TIED)) {
        if (verbose) Serial.printf("[Model] Unsupported image format (version %u).\n", h.version);
        return false;
    }
    if (h.hidden_dim != GEMMA_HIDDEN_DIM || h.intermediate_dim != GEMMA_INTERMEDIATE_DIM ||
        h.num_heads != GEMMA_NUM_HEADS || h.num_layers != TOTAL_TRANSFORMER_LAYERS ||
        h.vocab_size != GEMMA_VOCAB_SIZE || h.payload_bytes != GEMMA_MODEL_PAYLOAD_BYTES) {
        if (verbose) {
            Serial.printf("[Model] Image shape H=%u I=%u heads=%u layers=%u vocab=%u doesn't match the firmware "
                          "(H=%d I=%d heads=%d layers=%d vocab=%d).\n",
                          (unsigned)h.hidden_dim, (unsigned)h.intermediate_dim, (unsigned)h.num_heads,
                          (unsigned)h.num_layers, (unsigned)h.vocab_size, GEMMA_HIDDEN_DIM, GEMMA_INTERMEDIATE_DIM,
                          GEMMA_NUM_HEADS, TOTAL_TRANSFORMER_LAYERS, GEMMA_VOCAB_SIZE);
        }
        return false;
    }
    return true;
}

// Copies the image at base_addr into PSRAM, checks its CRC, and maps the layout.
// If out_sha256 is given, it receives the SHA-256 of the whole image (header + payload).
static bool gemma_load_model(uint32_t base_addr, uint32_t max_bytes, uint8_t out_sha256[32] = nullptr) {
    gemma_unload_model();
    if (!g_w25) {
        Serial.println("[Model] No W25Q128 detected; nothing to load.");
        return false;
    }

    W25ModelHeader h;
    if (!gemma_read_image_header(base_addr, h)) return false;
    if (W25_MODEL_HEADER_BYTES + h.payload_bytes > max_bytes || base_addr + W25_MODEL_HEADER_BYTES + h.payload_bytes > g_w25_size) {
        Serial.println("[Model] Image is larger than its flash region.");
        return false;
    }

    g_model_blob = (uint8_t*)heap_caps_malloc(h.payload_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!g_model_blob) {
        Serial.printf("[Model] Not enough PSRAM: need %u bytes, largest free block is %u.\n",
                      (unsigned)h.payload_bytes, (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
        return false;
    }

    Serial.printf("[Model] Copying %.2f MB from W25Q128 @0x%06X into PSRAM", h.payload_bytes / (1024.0f * 1024.0f), (unsigned)base_addr);
    Sha256 sha;
    sha.update(&h, sizeof(h));
    const uint32_t CHUNK = 32 * 1024;
    uint32_t start = millis();
    uint32_t crc = 0;
    for (uint32_t off = 0; off < h.payload_bytes; off += CHUNK) {
        uint32_t n = min(CHUNK, h.payload_bytes - off);
        if (!w25_read(base_addr + W25_MODEL_HEADER_BYTES + off, g_model_blob + off, n)) {
            Serial.printf("\n[Model] Read failed at offset %u.\n", (unsigned)off);
            gemma_unload_model();
            return false;
        }
        crc = crc32_update(crc, g_model_blob + off, n);
        sha.update(g_model_blob + off, n);
        if ((off / CHUNK) % 32 == 0) Serial.print(".");
    }
    if (crc != h.payload_crc32) {
        Serial.printf("\n[Model] CRC mismatch (read %08X, image says %08X).\n", (unsigned)crc, (unsigned)h.payload_crc32);
        gemma_unload_model();
        return false;
    }
    if (out_sha256) sha.finish(out_sha256);

    gemma_map_layout(g_model_blob);
    g_model.model_version = h.model_version;
    g_model.loaded = true;
    Serial.printf(" done in %u ms (model v%u, CRC %08X).\n", (unsigned)(millis() - start), (unsigned)h.model_version, (unsigned)crc);
    return true;
}

// ==============================================================================
// Tokenizer helpers
// ==============================================================================
static int gemma_tokenize(const char* text, uint32_t* out, int max_tokens) {
    if (!text || !*text || max_tokens <= 0) return 0;
    if (g_tokenizer && g_tokenizer->initialized) {
        return g_tokenizer->encode(String(text), out, max_tokens);
    }
    // No tokenizer partition: fall back to byte tokens <0x00>..<0xFF>
    int n = 0;
    for (const char* p = text; *p && n < max_tokens; p++) {
        out[n++] = GEMMA_BYTE_TOKEN_OFFSET + (uint8_t)*p;
    }
    return n;
}

static String gemma_decode_token(uint32_t token_id) {
    if (g_tokenizer && g_tokenizer->initialized) return g_tokenizer->decode(token_id);
    if (token_id >= GEMMA_BYTE_TOKEN_OFFSET && token_id < GEMMA_BYTE_TOKEN_OFFSET + 256) {
        return String((char)(token_id - GEMMA_BYTE_TOKEN_OFFSET));
    }
    return String();
}

// ==============================================================================
// Repetition Penalty & Top-P Sampler
// ==============================================================================
static void gemma_reset_recent_tokens() { g_recent_token_count = 0; }

static void update_recent_tokens(uint32_t token) {
    if (g_recent_token_count < REPETITION_WINDOW) {
        g_recent_tokens[g_recent_token_count++] = token;
    } else {
        memmove(&g_recent_tokens[0], &g_recent_tokens[1], (REPETITION_WINDOW - 1) * sizeof(uint32_t));
        g_recent_tokens[REPETITION_WINDOW - 1] = token;
    }
}

static int compare_prob_desc(const void* a, const void* b) {
    float pa = ((const ProbIndex*)a)->p;
    float pb = ((const ProbIndex*)b)->p;
    return (pa < pb) - (pa > pb);
}

static inline float random_unit() {
    return (esp_random() >> 8) * (1.0f / 16777216.0f);   // [0, 1)
}

static uint32_t argmax_token(const float* logits, int vocab_size) {
    int best = 0;
    for (int i = 1; i < vocab_size; i++) if (logits[i] > logits[best]) best = i;
    return (uint32_t)best;
}

// Nucleus sampling: sample from the smallest set of most-likely tokens whose
// probabilities add up to at least top_p.
static uint32_t sample_top_p(float* logits, int vocab_size, float temperature, float top_p) {
    // 1. Repetition penalty, once per distinct recent token
    for (int i = 0; i < g_recent_token_count; i++) {
        uint32_t tok = g_recent_tokens[i];
        if (tok >= (uint32_t)vocab_size) continue;
        bool already_penalized = false;
        for (int j = 0; j < i; j++) {
            if (g_recent_tokens[j] == tok) {
                already_penalized = true;
                break;
            }
        }
        if (already_penalized) continue;
        if (logits[tok] > 0.0f) logits[tok] /= REPETITION_PENALTY;
        else logits[tok] *= REPETITION_PENALTY;
    }

    // 2. Temperature + softmax
    float inv_temp = 1.0f / fmaxf(temperature, 0.01f);
    float max_l = -1e30f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
        if (logits[i] > max_l) max_l = logits[i];
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_l);
        sum_exp += logits[i];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < vocab_size; i++) logits[i] *= inv_sum;

    // 3. Candidates sorted by probability. A token below (1 - top_p) / (n - 1)
    //    can never be inside the nucleus, so skip those before sorting.
    const float cutoff = (1.0f - top_p) / (float)(vocab_size - 1);
    int n = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] >= cutoff) {
            g_candidates[n].p = logits[i];
            g_candidates[n].id = (uint16_t)i;
            n++;
        }
    }
    if (n == 0) return argmax_token(logits, vocab_size);
    qsort(g_candidates, n, sizeof(ProbIndex), compare_prob_desc);

    // 4. Cut at top_p, then sample within the nucleus (renormalized)
    float nucleus_mass = 0.0f;
    int last = n - 1;
    for (int i = 0; i < n; i++) {
        nucleus_mass += g_candidates[i].p;
        if (nucleus_mass >= top_p) {
            last = i;
            break;
        }
    }
    float r = random_unit() * nucleus_mass;
    float acc = 0.0f;
    for (int i = 0; i <= last; i++) {
        acc += g_candidates[i].p;
        if (r < acc) return g_candidates[i].id;
    }
    return g_candidates[last].id;
}

// ==============================================================================
// Transformer Pipeline: embed -> layers -> classify
// ==============================================================================
static void embed_token(uint32_t token_id, float* out_activation) {
    g_model.token_table.dequantize_embedding(token_id, out_activation);
}

static void run_layer(int l, float* activation, int pos) {
    const GemmaInt16LayerWeights& lw = g_model.layers[l];

    // --- 1. Multi-Head Self-Attention Block ---
    gemma_rmsnorm(g_norm_buf, activation, lw.attn_norm, GEMMA_HIDDEN_DIM);
    lw.wq.matmul_parallel(g_q, g_norm_buf);
    lw.wk.matmul_parallel(g_k, g_norm_buf);
    lw.wv.matmul_parallel(g_v, g_norm_buf);
    gemma_apply_rope(g_q, g_k, pos);
    g_cache_mgr.kv.store(l, pos, g_k, g_v);

    int valid_ctx = min(pos + 1, (int)g_cache_mgr.kv.max_context);
    for (int h = 0; h < GEMMA_NUM_HEADS; h++) {
        const float* q_h = &g_q[h * GEMMA_HEAD_DIM];
        float max_score = -1e30f;
        float scores[GEMMA_MAX_CONTEXT_LENGTH];

        for (int t = 0; t < valid_ctx; t++) {
            const float* k_h = g_cache_mgr.kv.get_k(l, t) + (h * GEMMA_HEAD_DIM);
            float dot = 0.0f;
            for (int d = 0; d < GEMMA_HEAD_DIM; d++) dot += q_h[d] * k_h[d];
            dot *= GEMMA_QUERY_SCALE;
            scores[t] = dot;
            if (dot > max_score) max_score = dot;
        }

        float sum_exp = 0.0f;
        for (int t = 0; t < valid_ctx; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp += scores[t];
        }
        float inv_sum = 1.0f / sum_exp;

        float* head_out = &g_attn_output[h * GEMMA_HEAD_DIM];
        for (int d = 0; d < GEMMA_HEAD_DIM; d++) head_out[d] = 0.0f;
        for (int t = 0; t < valid_ctx; t++) {
            float weight = scores[t] * inv_sum;
            const float* v_h = g_cache_mgr.kv.get_v(l, t) + (h * GEMMA_HEAD_DIM);
            for (int d = 0; d < GEMMA_HEAD_DIM; d++) head_out[d] += weight * v_h[d];
        }
    }

    lw.wo.matmul_parallel(g_proj_output, g_attn_output);
    for (int i = 0; i < GEMMA_HIDDEN_DIM; i++) activation[i] += g_proj_output[i];

    // --- 2. Feed-Forward Network (GeGLU) Block ---
    gemma_rmsnorm(g_norm_buf, activation, lw.ffn_norm, GEMMA_HIDDEN_DIM);
    lw.w_gate.matmul_parallel(g_h1_gate, g_norm_buf);
    lw.w_up.matmul_parallel(g_h3_up, g_norm_buf);
    gemma_geglu(g_geglu_buf, g_h1_gate, g_h3_up, GEMMA_INTERMEDIATE_DIM);
    lw.w_down.matmul_parallel(g_h2_down, g_geglu_buf);
    for (int i = 0; i < GEMMA_HIDDEN_DIM; i++) activation[i] += g_h2_down[i];
}

// Runs one token at `pos` through every layer; leaves the result in g_activation.
static void forward_token(uint32_t token_id, int pos) {
    embed_token(token_id, g_activation);
    for (int l = 0; l < TOTAL_TRANSFORMER_LAYERS; l++) {
        run_layer(l, g_activation, pos);
    }
}

// Runs tokens[from..to) at positions from..to-1 (fills the KV cache only).
static void prefill(const uint32_t* tokens, int from, int to) {
    for (int i = from; i < to; i++) forward_token(tokens[i], i);
}

// Final RMSNorm, tied classifier, then sampling. temperature <= 0 means greedy
// (argmax, no repetition penalty), which the edge build uses for actions so
// that the same prompt always gives the same output.
static uint32_t sample_next_token(float temperature = DEFAULT_TEMPERATURE, float top_p = DEFAULT_TOP_P) {
    gemma_rmsnorm(g_norm_buf, g_activation, g_model.final_norm, GEMMA_HIDDEN_DIM);
    g_model.token_table.as_classifier().matmul_parallel(g_logits, g_norm_buf);
    uint32_t next_token = (temperature <= 0.0f) ? argmax_token(g_logits, GEMMA_VOCAB_SIZE)
                                                : sample_top_p(g_logits, GEMMA_VOCAB_SIZE, temperature, top_p);
    update_recent_tokens(next_token);
    return next_token;
}

// Greedy generation from [prompt]: writes up to max_new tokens (EOS excluded) to out.
// Returns the number generated. The KV cache is overwritten from position 0.
static int gemma_generate_greedy(const uint32_t* prompt, int n_prompt, uint32_t* out, int max_new) {
    if (!g_model.loaded || n_prompt <= 0 || n_prompt > GEMMA_MAX_CONTEXT_LENGTH) return 0;
    gemma_reset_recent_tokens();
    prefill(prompt, 0, n_prompt - 1);
    uint32_t current = prompt[n_prompt - 1];
    int pos = n_prompt - 1;
    int n = 0;
    while (n < max_new && pos < GEMMA_MAX_CONTEXT_LENGTH) {
        forward_token(current, pos);
        uint32_t next = sample_next_token(0.0f);
        if (next == GEMMA_EOS_TOKEN_ID) break;
        out[n++] = next;
        current = next;
        pos++;
    }
    return n;
}
