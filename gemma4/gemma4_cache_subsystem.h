#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <Arduino.h>
#include "esp_heap_caps.h"
#include "gemma4_w25q128_config.h"

// ==============================================================================
// Gemma 4 Advanced Multi-Tier Caching Subsystem for ESP32-S3
// ==============================================================================
// 1. KV Cache:      Dense multi-head key-value state across all 18 layers
// 2. Prefix Cache:  Fast checkpoint restore for common system prompts / prefixes
// 3. Prompt Cache:  64-bit FNV-1a exact match hash cache
// 4. Semantic Cache: Dense centroid cosine-similarity matching (threshold >= 0.90)
// ==============================================================================

#define CACHE_MAX_PREFIX_TOKENS      32
#define CACHE_MAX_PREFIX_ENTRIES      4
#define CACHE_PROMPT_ENTRIES          8
#define CACHE_SEMANTIC_ENTRIES        8
#define CACHE_MAX_RESPONSE_LEN       256
#define CACHE_SEMANTIC_THRESHOLD     0.90f

// Large cache buffers go to PSRAM when there is one.
static inline void* cache_alloc(size_t bytes) {
#if defined(BOARD_HAS_PSRAM)
    void* p = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (p) return p;
#endif
    return malloc(bytes);
}

// ==============================================================================
// 1. KV CACHE: Multi-Layer, Multi-Head Key-Value Cache
// ==============================================================================
class KVCache {
public:
    uint16_t current_seq_len;
    uint16_t max_context;

    // Pointers for each layer (18 layers total)
    // Layout: [layer][pos * GEMMA_HIDDEN_DIM]
    float* k_data[TOTAL_TRANSFORMER_LAYERS];
    float* v_data[TOTAL_TRANSFORMER_LAYERS];
    bool allocated;

    KVCache() : current_seq_len(0), max_context(GEMMA_MAX_CONTEXT_LENGTH), allocated(false) {
        for (int l = 0; l < TOTAL_TRANSFORMER_LAYERS; l++) {
            k_data[l] = nullptr;
            v_data[l] = nullptr;
        }
    }

    bool init(uint16_t ctx_len = GEMMA_MAX_CONTEXT_LENGTH) {
        max_context = ctx_len;
        size_t layer_bytes = max_context * GEMMA_HIDDEN_DIM * sizeof(float);

        for (int l = 0; l < TOTAL_TRANSFORMER_LAYERS; l++) {
            k_data[l] = (float*)cache_alloc(layer_bytes);
            v_data[l] = (float*)cache_alloc(layer_bytes);

            if (!k_data[l] || !v_data[l]) {
                Serial.printf("[KVCache] Allocation failed at layer %d!\n", l);
                return false;
            }
            memset(k_data[l], 0, layer_bytes);
            memset(v_data[l], 0, layer_bytes);
        }
        current_seq_len = 0;
        allocated = true;
        return true;
    }

    inline void wipe() {
        current_seq_len = 0;
        size_t layer_bytes = max_context * GEMMA_HIDDEN_DIM * sizeof(float);
        for (int l = 0; l < TOTAL_TRANSFORMER_LAYERS; l++) {
            if (k_data[l]) memset(k_data[l], 0, layer_bytes);
            if (v_data[l]) memset(v_data[l], 0, layer_bytes);
        }
    }

    inline void store(int layer, int pos, const float* k_vec, const float* v_vec) {
        if (pos < 0 || pos >= max_context || layer >= TOTAL_TRANSFORMER_LAYERS) return;
        int offset = pos * GEMMA_HIDDEN_DIM;
        memcpy(&k_data[layer][offset], k_vec, GEMMA_HIDDEN_DIM * sizeof(float));
        memcpy(&v_data[layer][offset], v_vec, GEMMA_HIDDEN_DIM * sizeof(float));
        if (pos + 1 > current_seq_len) current_seq_len = pos + 1;
    }

    inline const float* get_k(int layer, int pos) const {
        return &k_data[layer][pos * GEMMA_HIDDEN_DIM];
    }

    inline const float* get_v(int layer, int pos) const {
        return &v_data[layer][pos * GEMMA_HIDDEN_DIM];
    }
};

// ==============================================================================
// 2. PREFIX CACHE: Common System Prompt & Instruction Checkpointing
// ==============================================================================
struct PrefixCacheEntry {
    bool active;
    uint16_t prefix_len;
    uint16_t capacity;        // tokens the k/v buffers below can hold
    uint32_t tokens[CACHE_MAX_PREFIX_TOKENS];
    char description[48];

    // Snapshot of precomputed K and V states for the prefix tokens
    float* k_prefix[TOTAL_TRANSFORMER_LAYERS];
    float* v_prefix[TOTAL_TRANSFORMER_LAYERS];
};

class PrefixCache {
public:
    PrefixCacheEntry entries[CACHE_MAX_PREFIX_ENTRIES];
    uint32_t total_hits;
    uint32_t total_lookups;

    PrefixCache() : total_hits(0), total_lookups(0) {
        for (int i = 0; i < CACHE_MAX_PREFIX_ENTRIES; i++) {
            entries[i].active = false;
            entries[i].prefix_len = 0;
            entries[i].capacity = 0;
            entries[i].description[0] = '\0';
            for (int l = 0; l < TOTAL_TRANSFORMER_LAYERS; l++) {
                entries[i].k_prefix[l] = nullptr;
                entries[i].v_prefix[l] = nullptr;
            }
        }
    }

    // Snapshot KV positions [0, len) of every layer into `slot`.
    // Precondition: `tokens` were just run through the model at positions 0..len-1,
    // so source_kv holds their K/V. A snapshot of anything else is meaningless.
    bool register_prefix(int slot, const char* desc, const uint32_t* tokens, uint16_t len, const KVCache& source_kv) {
        if (slot < 0 || slot >= CACHE_MAX_PREFIX_ENTRIES) return false;
        if (len == 0 || len > CACHE_MAX_PREFIX_TOKENS || len > source_kv.max_context) return false;

        PrefixCacheEntry& e = entries[slot];
        e.active = false;

        // Buffers sized for a shorter prefix can't hold this one: reallocate.
        if (len > e.capacity) {
            release(e);
            size_t cap_bytes = (size_t)len * GEMMA_HIDDEN_DIM * sizeof(float);
            for (int l = 0; l < TOTAL_TRANSFORMER_LAYERS; l++) {
                e.k_prefix[l] = (float*)cache_alloc(cap_bytes);
                e.v_prefix[l] = (float*)cache_alloc(cap_bytes);
                if (!e.k_prefix[l] || !e.v_prefix[l]) {
                    release(e);
                    return false;
                }
            }
            e.capacity = len;
        }

        size_t bytes = (size_t)len * GEMMA_HIDDEN_DIM * sizeof(float);
        for (int l = 0; l < TOTAL_TRANSFORMER_LAYERS; l++) {
            memcpy(e.k_prefix[l], source_kv.k_data[l], bytes);
            memcpy(e.v_prefix[l], source_kv.v_data[l], bytes);
        }
        e.prefix_len = len;
        memcpy(e.tokens, tokens, len * sizeof(uint32_t));
        strncpy(e.description, desc, sizeof(e.description) - 1);
        e.description[sizeof(e.description) - 1] = '\0';
        e.active = true;
        return true;
    }

    // Deactivate every slot (buffers are kept for reuse).
    void clear() {
        for (int i = 0; i < CACHE_MAX_PREFIX_ENTRIES; i++) entries[i].active = false;
    }

    static void release(PrefixCacheEntry& e) {
        for (int l = 0; l < TOTAL_TRANSFORMER_LAYERS; l++) {
            free(e.k_prefix[l]);
            free(e.v_prefix[l]);
            e.k_prefix[l] = nullptr;
            e.v_prefix[l] = nullptr;
        }
        e.capacity = 0;
    }

    // Matches incoming prompt tokens against registered prefix entries
    int match(const uint32_t* prompt_tokens, uint16_t prompt_len, int& matched_slot) {
        total_lookups++;
        int best_len = 0;
        matched_slot = -1;

        for (int i = 0; i < CACHE_MAX_PREFIX_ENTRIES; i++) {
            if (!entries[i].active) continue;
            if (prompt_len < entries[i].prefix_len) continue;

            bool match = true;
            for (int t = 0; t < entries[i].prefix_len; t++) {
                if (prompt_tokens[t] != entries[i].tokens[t]) {
                    match = false;
                    break;
                }
            }

            if (match && entries[i].prefix_len > best_len) {
                best_len = entries[i].prefix_len;
                matched_slot = i;
            }
        }

        if (best_len > 0) total_hits++;
        return best_len;
    }

    // Restores cached prefix KV states into active KV cache
    void restore(int slot, KVCache& dest_kv) {
        if (slot < 0 || slot >= CACHE_MAX_PREFIX_ENTRIES || !entries[slot].active) return;
        const PrefixCacheEntry& e = entries[slot];
        size_t bytes = e.prefix_len * GEMMA_HIDDEN_DIM * sizeof(float);

        for (int l = 0; l < TOTAL_TRANSFORMER_LAYERS; l++) {
            if (e.k_prefix[l] && dest_kv.k_data[l]) {
                memcpy(dest_kv.k_data[l], e.k_prefix[l], bytes);
            }
            if (e.v_prefix[l] && dest_kv.v_data[l]) {
                memcpy(dest_kv.v_data[l], e.v_prefix[l], bytes);
            }
        }
        dest_kv.current_seq_len = e.prefix_len;
    }
};

// ==============================================================================
// 3. PROMPT CACHE: 64-Bit FNV-1a Exact Prompt Hash Cache
// ==============================================================================
struct PromptCacheEntry {
    bool active;
    uint64_t hash;
    uint16_t prompt_len;
    uint32_t timestamp;
    char response[CACHE_MAX_RESPONSE_LEN];
};

class PromptCache {
public:
    PromptCacheEntry entries[CACHE_PROMPT_ENTRIES];
    uint32_t total_hits;
    uint32_t total_lookups;

    PromptCache() : total_hits(0), total_lookups(0) {
        for (int i = 0; i < CACHE_PROMPT_ENTRIES; i++) {
            entries[i].active = false;
            entries[i].hash = 0;
            entries[i].response[0] = '\0';
        }
    }

    // 64-bit FNV-1a hash over token sequence
    static uint64_t compute_hash(const uint32_t* tokens, uint16_t len) {
        uint64_t h = 14695981039346656037ULL;
        for (uint16_t i = 0; i < len; i++) {
            h ^= (uint64_t)tokens[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    // Exact match lookup
    bool lookup(uint64_t hash, char* out_response, size_t max_out_len) {
        total_lookups++;
        for (int i = 0; i < CACHE_PROMPT_ENTRIES; i++) {
            if (entries[i].active && entries[i].hash == hash) {
                total_hits++;
                strncpy(out_response, entries[i].response, max_out_len - 1);
                out_response[max_out_len - 1] = '\0';
                entries[i].timestamp = millis();
                return true;
            }
        }
        return false;
    }

    // Store prompt and generated response with LRU eviction
    void store(uint64_t hash, uint16_t prompt_len, const char* response) {
        int target_slot = 0;
        uint32_t oldest_time = 0xFFFFFFFF;

        for (int i = 0; i < CACHE_PROMPT_ENTRIES; i++) {
            if (!entries[i].active) {
                target_slot = i;
                break;
            }
            if (entries[i].timestamp < oldest_time) {
                oldest_time = entries[i].timestamp;
                target_slot = i;
            }
        }

        PromptCacheEntry& e = entries[target_slot];
        e.active = true;
        e.hash = hash;
        e.prompt_len = prompt_len;
        e.timestamp = millis();
        strncpy(e.response, response, sizeof(e.response) - 1);
        e.response[sizeof(e.response) - 1] = '\0';
    }

    void clear() {
        for (int i = 0; i < CACHE_PROMPT_ENTRIES; i++) entries[i].active = false;
    }
};

// ==============================================================================
// 4. SEMANTIC CACHE: Embedding Centroid & Cosine Similarity Matcher
// ==============================================================================
struct SemanticCacheEntry {
    bool active;
    uint32_t timestamp;
    float centroid[GEMMA_HIDDEN_DIM]; // Unit-normalized prompt centroid vector
    char query[64];
    char response[CACHE_MAX_RESPONSE_LEN];
};

class SemanticCache {
public:
    SemanticCacheEntry entries[CACHE_SEMANTIC_ENTRIES];
    uint32_t total_hits;
    uint32_t total_lookups;

    SemanticCache() : total_hits(0), total_lookups(0) {
        for (int i = 0; i < CACHE_SEMANTIC_ENTRIES; i++) {
            entries[i].active = false;
            entries[i].timestamp = 0;
            entries[i].query[0] = '\0';
            entries[i].response[0] = '\0';
        }
    }

    // Scales a summed embedding vector to unit length, so cosine similarity is a dot
    // product. (Sum and mean give the same direction, so the caller can just add
    // token embeddings into one buffer instead of storing all of them.)
    static void normalize(float* centroid) {
        float norm_sq = 0.0f;
        for (int i = 0; i < GEMMA_HIDDEN_DIM; i++) norm_sq += centroid[i] * centroid[i];
        float inv_norm = 1.0f / sqrtf(norm_sq + 1e-9f);
        for (int i = 0; i < GEMMA_HIDDEN_DIM; i++) centroid[i] *= inv_norm;
    }

    void clear() {
        for (int i = 0; i < CACHE_SEMANTIC_ENTRIES; i++) entries[i].active = false;
    }

    // Cosine similarity dot product between two unit-normalized vectors
    static inline float cosine_similarity(const float* a, const float* b) {
        float dot = 0.0f;
        for (int i = 0; i < GEMMA_HIDDEN_DIM; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }

    // Semantic lookup by cosine distance threshold
    bool lookup(const float* query_centroid, float threshold, char* out_response, size_t max_out_len, float& out_sim, char* matched_query = nullptr) {
        total_lookups++;
        float best_sim = -1.0f;
        int best_slot = -1;

        for (int i = 0; i < CACHE_SEMANTIC_ENTRIES; i++) {
            if (!entries[i].active) continue;
            float sim = cosine_similarity(query_centroid, entries[i].centroid);
            if (sim > best_sim) {
                best_sim = sim;
                best_slot = i;
            }
        }

        out_sim = best_sim;
        if (best_slot >= 0 && best_sim >= threshold) {
            total_hits++;
            entries[best_slot].timestamp = millis();
            strncpy(out_response, entries[best_slot].response, max_out_len - 1);
            out_response[max_out_len - 1] = '\0';
            if (matched_query) {
                strncpy(matched_query, entries[best_slot].query, 63);
                matched_query[63] = '\0';
            }
            return true;
        }
        return false;
    }

    // Store query centroid and response with LRU replacement
    void store(const float* centroid, const char* query, const char* response) {
        int target_slot = 0;
        uint32_t oldest_time = 0xFFFFFFFF;

        for (int i = 0; i < CACHE_SEMANTIC_ENTRIES; i++) {
            if (!entries[i].active) {
                target_slot = i;
                break;
            }
            if (entries[i].timestamp < oldest_time) {
                oldest_time = entries[i].timestamp;
                target_slot = i;
            }
        }

        SemanticCacheEntry& e = entries[target_slot];
        e.active = true;
        e.timestamp = millis();
        memcpy(e.centroid, centroid, GEMMA_HIDDEN_DIM * sizeof(float));
        strncpy(e.query, query, sizeof(e.query) - 1);
        e.query[sizeof(e.query) - 1] = '\0';
        strncpy(e.response, response, sizeof(e.response) - 1);
        e.response[sizeof(e.response) - 1] = '\0';
    }
};

// ==============================================================================
// Unified Cache Manager
// ==============================================================================
struct UnifiedCacheManager {
    KVCache kv;
    PrefixCache prefix;
    PromptCache prompt;
    SemanticCache semantic;

    void print_stats() const {
        Serial.println("\n=== Gemma 4 Multi-Tier Cache Telemetry ===");
        Serial.printf(" [KV Cache]       Seq Len: %d / %d tokens\n", kv.current_seq_len, kv.max_context);
        Serial.printf(" [Prefix Cache]   Hits: %u | Lookups: %u (%.1f%%)\n", 
                      prefix.total_hits, prefix.total_lookups, 
                      prefix.total_lookups > 0 ? (100.0f * prefix.total_hits / prefix.total_lookups) : 0.0f);
        Serial.printf(" [Prompt Cache]   Hits: %u | Lookups: %u (%.1f%%)\n", 
                      prompt.total_hits, prompt.total_lookups, 
                      prompt.total_lookups > 0 ? (100.0f * prompt.total_hits / prompt.total_lookups) : 0.0f);
        Serial.printf(" [Semantic Cache] Hits: %u | Lookups: %u (%.1f%%)\n", 
                      semantic.total_hits, semantic.total_lookups, 
                      semantic.total_lookups > 0 ? (100.0f * semantic.total_hits / semantic.total_lookups) : 0.0f);
        Serial.println("==========================================\n");
    }
};
