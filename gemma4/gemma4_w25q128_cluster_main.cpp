#include "gemma4_engine.h"

// ==============================================================================
// Gemma 4 Single ESP32-S3 + External W25Q128: Serial Chat Build
// ==============================================================================
// - Storage: W25Q128 (16 MB SPI NOR) wired to SPI2; one model image at address 0,
//   copied into PSRAM at boot (see gemma4_engine.h).
// - Caching: KV cache, prefix cache (system prompt), prompt cache, semantic cache
// For the self-evolving edge node (A/B model slots, OTA, UART discovery) see edge/.
// ==============================================================================

// Prompt state
static const int kMaxInputChars = 128;
static char g_system_prompt[96] = GEMMA_DEFAULT_SYSTEM_PROMPT;
static uint32_t g_prefix_tokens[CACHE_MAX_PREFIX_TOKENS];   // [BOS] + system prompt tokens
static int g_prefix_len = 0;
static uint32_t g_prompt_tokens[GEMMA_MAX_CONTEXT_LENGTH];
static uint32_t g_scratch_tokens[kMaxInputChars];
static float g_query_centroid[GEMMA_HIDDEN_DIM];
static char g_response_accumulator[CACHE_MAX_RESPONSE_LEN];
static uint16_t g_response_acc_len = 0;

static bool load_model_from_w25() { return gemma_load_model(0, g_w25_size); }

// ==============================================================================
// System Prompt (Prefix Cache Slot 0)
// ==============================================================================
// Runs [BOS] + system prompt through the model once and snapshots the resulting
// KV state, so every user prompt can skip that part of the prefill.
static bool set_system_prompt(const char* text) {
    if (text != g_system_prompt) {
        strncpy(g_system_prompt, text, sizeof(g_system_prompt) - 1);
        g_system_prompt[sizeof(g_system_prompt) - 1] = '\0';
    }
    if (!g_model.loaded) return false;   // applied once a model is loaded

    int n = gemma_tokenize(g_system_prompt, g_scratch_tokens, kMaxInputChars);
    if (1 + n > CACHE_MAX_PREFIX_TOKENS) {
        Serial.printf("[System] System prompt is %d tokens; the prefix cache holds %d. Using no system prompt.\n",
                      n, CACHE_MAX_PREFIX_TOKENS - 1);
        g_system_prompt[0] = '\0';
        n = 0;
    }
    g_prefix_tokens[0] = GEMMA_BOS_TOKEN_ID;
    memcpy(&g_prefix_tokens[1], g_scratch_tokens, n * sizeof(uint32_t));
    g_prefix_len = 1 + n;

    g_cache_mgr.kv.wipe();
    prefill(g_prefix_tokens, 0, g_prefix_len);
    if (!g_cache_mgr.prefix.register_prefix(0, "System prompt", g_prefix_tokens, g_prefix_len, g_cache_mgr.kv)) {
        Serial.println("[Prefix Cache] Could not store the system prompt KV state (out of memory?).");
        return false;
    }
    // Cached replies were produced under the old system prompt.
    g_cache_mgr.prompt.clear();
    g_cache_mgr.semantic.clear();

    Serial.printf("[Prefix Cache] System prompt \"%s\" (%d tokens incl. BOS) cached in slot 0.\n",
                  g_system_prompt, g_prefix_len);
    return true;
}

// ==============================================================================
// Prompt Execution Engine with Multi-Tier Cache Verification
// ==============================================================================
static void process_user_prompt(const char* prompt_text) {
    if (!prompt_text || !*prompt_text) return;
    if (!g_model.loaded) {
        Serial.println("[Model] No model loaded. Upload one with gemma4/w25q128_upload.py (/help for commands).");
        return;
    }

    Serial.printf(">>> Prompt: \"%s\"\n", prompt_text);

    // 1. Tokens = [BOS] + system prompt + user text
    memcpy(g_prompt_tokens, g_prefix_tokens, g_prefix_len * sizeof(uint32_t));
    int n_user = gemma_tokenize(prompt_text, g_scratch_tokens, kMaxInputChars);
    int room = GEMMA_MAX_CONTEXT_LENGTH - g_prefix_len;
    if (n_user > room) {
        Serial.printf("[Tokenizer] Prompt truncated from %d to %d tokens to fit the context.\n", n_user, room);
        n_user = room;
    }
    if (n_user <= 0) return;
    memcpy(&g_prompt_tokens[g_prefix_len], g_scratch_tokens, n_user * sizeof(uint32_t));
    const int num_tokens = g_prefix_len + n_user;
    Serial.printf("[Tokenizer] %d tokens (%d system + %d prompt).\n", num_tokens, g_prefix_len, n_user);

    // CACHE TIER 1: PROMPT CACHE (Exact FNV-1a Hash Match)
    uint64_t prompt_hash = PromptCache::compute_hash(g_prompt_tokens, num_tokens);
    char cached_resp[CACHE_MAX_RESPONSE_LEN];
    if (g_cache_mgr.prompt.lookup(prompt_hash, cached_resp, sizeof(cached_resp))) {
        Serial.println("\n------------------------------------------------------------");
        Serial.printf("[PROMPT CACHE HIT] Exact match (Hash: 0x%016llX)\n", prompt_hash);
        Serial.println(" -> Skipped prefill and generation.");
        Serial.println("------------------------------------------------------------");
        Serial.printf(">>> Output: %s\n", cached_resp);
        return;
    }

    // CACHE TIER 2: SEMANTIC CACHE (cosine similarity of the user tokens' mean
    // embedding; the system prompt is left out so it doesn't dominate)
    memset(g_query_centroid, 0, sizeof(g_query_centroid));
    for (int t = g_prefix_len; t < num_tokens; t++) {
        embed_token(g_prompt_tokens[t], g_norm_buf);
        for (int i = 0; i < GEMMA_HIDDEN_DIM; i++) g_query_centroid[i] += g_norm_buf[i];
    }
    SemanticCache::normalize(g_query_centroid);

    float sim_score = 0.0f;
    char matched_query[64];
    if (g_cache_mgr.semantic.lookup(g_query_centroid, CACHE_SEMANTIC_THRESHOLD, cached_resp, sizeof(cached_resp), sim_score, matched_query)) {
        Serial.println("\n------------------------------------------------------------");
        Serial.printf("[SEMANTIC CACHE HIT] Cosine Similarity: %.4f (Threshold: %.2f)\n", sim_score, CACHE_SEMANTIC_THRESHOLD);
        Serial.printf(" -> Matched query: \"%s\"\n", matched_query);
        Serial.println("------------------------------------------------------------");
        Serial.printf(">>> Output: %s\n", cached_resp);
        return;
    }

    // CACHE TIER 3: PREFIX CACHE (system prompt KV state)
    int matched_slot = -1;
    int matched_len = g_cache_mgr.prefix.match(g_prompt_tokens, num_tokens, matched_slot);
    int start_pos = 0;
    if (matched_len > 0) {
        g_cache_mgr.prefix.restore(matched_slot, g_cache_mgr.kv);
        // The last prompt token always runs in the generation loop (it produces the
        // first logits), so never skip it even if the whole prompt is cached.
        start_pos = min(matched_len, num_tokens - 1);
        Serial.printf("[PREFIX CACHE HIT] Restored %d tokens from slot %d (\"%s\"); prefill starts at position %d.\n",
                      matched_len, matched_slot, g_cache_mgr.prefix.entries[matched_slot].description, start_pos);
    } else {
        g_cache_mgr.kv.wipe();
    }

    // TRANSFORMER PREFILL (all prompt tokens except the last)
    uint32_t prefill_start = millis();
    prefill(g_prompt_tokens, start_pos, num_tokens - 1);
    Serial.printf("[Prefill] %d tokens in %u ms.\n", max(0, num_tokens - 1 - start_pos), (unsigned)(millis() - prefill_start));

    // AUTOREGRESSIVE GENERATION LOOP (KV CACHE ACTIVE)
    Serial.print(">>> Output: ");
    g_response_acc_len = 0;
    g_response_accumulator[0] = '\0';
    gemma_reset_recent_tokens();

    uint32_t current_token = g_prompt_tokens[num_tokens - 1];
    int pos = num_tokens - 1;
    int generated_count = 0;
    uint32_t gen_start_time = millis();

    while (generated_count < MAX_NEW_TOKENS && pos < GEMMA_MAX_CONTEXT_LENGTH) {
        forward_token(current_token, pos);
        uint32_t next_token = sample_next_token();
        if (next_token == GEMMA_EOS_TOKEN_ID) break;

        String piece = gemma_decode_token(next_token);
        if (piece.length() > 0) {
            Serial.print(piece);
            size_t piece_len = piece.length();
            if (g_response_acc_len + piece_len < sizeof(g_response_accumulator) - 1) {
                memcpy(&g_response_accumulator[g_response_acc_len], piece.c_str(), piece_len + 1);
                g_response_acc_len += piece_len;
            }
        }

        current_token = next_token;
        pos++;
        generated_count++;
    }

    uint32_t total_gen_time = millis() - gen_start_time;
    float tok_per_sec = (generated_count > 0 && total_gen_time > 0) ? (generated_count * 1000.0f / total_gen_time) : 0.0f;
    Serial.printf("\n\n[Generation Finished] %d tokens in %u ms (%.2f tok/s)\n",
                  generated_count, (unsigned)total_gen_time, tok_per_sec);

    // POPULATE PROMPT & SEMANTIC CACHES FOR FUTURE CALLS
    if (g_response_acc_len > 0) {
        g_cache_mgr.prompt.store(prompt_hash, num_tokens, g_response_accumulator);
        g_cache_mgr.semantic.store(g_query_centroid, prompt_text, g_response_accumulator);
        Serial.println("[Cache Subsystem] Stored response in Prompt Cache & Semantic Cache.");
    }
}

// ==============================================================================
// Model (re)load and W25Q128 commands
// ==============================================================================
static bool reload_model() {
    if (!g_w25 && !w25_init()) return false;
    bool ok = load_model_from_w25();
    g_cache_mgr.kv.wipe();
    g_cache_mgr.prefix.clear();
    g_cache_mgr.prompt.clear();
    g_cache_mgr.semantic.clear();
    if (ok) set_system_prompt(g_system_prompt);
    return ok;
}

static void print_w25_info() {
    if (!g_w25) {
        Serial.println("[W25Q128] Not detected.");
        w25_print_wiring();
        return;
    }
    Serial.printf("[W25Q128] JEDEC ID 0x%06X, %u bytes, IO mode %d.\n", (unsigned)g_w25_id, (unsigned)g_w25_size, (int)W25_IO_MODE);
    W25ModelHeader h;
    if (!w25_read(0, &h, sizeof(h)) || h.magic != W25_MODEL_MAGIC) {
        Serial.println("[W25Q128] No model image on the chip.");
    } else {
        Serial.printf("[W25Q128] Image v%u (model v%u): H=%u I=%u heads=%u layers=%u vocab=%u, %u payload bytes, CRC %08X.\n",
                      h.version, (unsigned)h.model_version, (unsigned)h.hidden_dim, (unsigned)h.intermediate_dim,
                      (unsigned)h.num_heads, (unsigned)h.num_layers, (unsigned)h.vocab_size, (unsigned)h.payload_bytes,
                      (unsigned)h.payload_crc32);
    }
    Serial.printf("[Model] %s\n", g_model.loaded ? "Loaded in PSRAM." : "Not loaded.");
}

static void handle_w25write(const char* args) {
    unsigned long size = 0, crc = 0;
    if (sscanf(args, "%lu %lx", &size, &crc) != 2) {
        Serial.println("@W25 ERR usage: /w25write <bytes> <crc32 hex>");
        return;
    }
    // The PSRAM copy stops matching the chip as soon as erasing starts.
    gemma_unload_model();
    if (!w25_receive_image((uint32_t)size, (uint32_t)crc)) return;
    Serial.println(reload_model() ? "@W25 LOADED" : "@W25 ERR image written but the model did not load (see above)");
}

// ==============================================================================
// Serial CLI Command Dispatcher
// ==============================================================================
static void print_help() {
    Serial.println("\n--- Gemma 4 S3/W25Q128 CLI Commands ---");
    Serial.println(" <text>              Generate a reply (system prompt + text, multi-tier caching)");
    Serial.println(" /system             Show the system prompt");
    Serial.println(" /system <text>      Set the system prompt (its KV state is kept in the prefix cache)");
    Serial.println(" /system off         Use no system prompt");
    Serial.println(" /stats              Cache hit rates and memory use");
    Serial.println(" /clear              Wipe the active KV cache");
    Serial.println(" /w25info            W25Q128 chip ID and model image header");
    Serial.println(" /reload             Re-read the model from the W25Q128 into PSRAM");
    Serial.println(" /w25write <n> <crc> Receive a model image (used by gemma4/w25q128_upload.py)\n");
}

static void handle_cli_command(const char* cmd) {
    if (strcmp(cmd, "/help") == 0) {
        print_help();
    } else if (strcmp(cmd, "/stats") == 0) {
        g_cache_mgr.print_stats();
        Serial.printf(" [Memory] PSRAM free: %u KB | internal free: %u KB\n",
                      (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024),
                      (unsigned)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
    } else if (strcmp(cmd, "/clear") == 0) {
        g_cache_mgr.kv.wipe();
        Serial.println("[Cache] Active KV cache wiped.");
    } else if (strcmp(cmd, "/system") == 0) {
        Serial.printf("[System] \"%s\"\n", g_system_prompt);
    } else if (strcmp(cmd, "/system off") == 0) {
        set_system_prompt("");
    } else if (strncmp(cmd, "/system ", 8) == 0) {
        set_system_prompt(cmd + 8);
    } else if (strcmp(cmd, "/w25info") == 0) {
        print_w25_info();
    } else if (strcmp(cmd, "/reload") == 0) {
        reload_model();
    } else if (strncmp(cmd, "/w25write ", 10) == 0) {
        handle_w25write(cmd + 10);
    } else if (cmd[0] == '/') {
        Serial.println("Unknown command. Type /help.");
    } else {
        process_user_prompt(cmd);
    }
}

// ==============================================================================
// Arduino Setup & Main Loop
// ==============================================================================
void setup() {
    Serial.setRxBufferSize(2 * W25_UPLOAD_CHUNK);   // must precede begin(); /w25write streams 4 KB chunks
    Serial.begin(SERIAL_BAUD);
    delay(1500);

    Serial.println("\n===============================================================");
    Serial.println("   LISA - Gemma 4 Engine (ESP32-S3 + external W25Q128 on SPI2) ");
    Serial.println("   INT16 | Tied Embeddings | Dual-Core GEMM | 4-Tier Caching   ");
    Serial.println("===============================================================");

    gemma_start_gemm_worker();
    Serial.println("[Init] Dual-Core INT16 GEMM worker active on Core 0.");

    if (!psramFound()) {
        Serial.println("[FATAL] No PSRAM found. This build needs an ESP32-S3 with PSRAM (e.g. N16R8). Halting.");
        while (1) delay(1000);
    }

    if (gemma_init_tokenizer()) {
        Serial.printf("[Init] BPE tokenizer loaded (first %d tokens of tokenizer.bin).\n", GEMMA_VOCAB_SIZE);
    } else {
        Serial.println("[Init] Warning: 'tokenizer' partition missing or invalid. Using byte tokens.");
    }

    if (!g_cache_mgr.kv.init(GEMMA_MAX_CONTEXT_LENGTH)) {
        Serial.println("[FATAL] KV cache allocation failed. Halting.");
        while (1) delay(1000);
    }
    Serial.printf("[Init] KV cache: %d layers x %d positions.\n", TOTAL_TRANSFORMER_LAYERS, GEMMA_MAX_CONTEXT_LENGTH);

    if (w25_init() && load_model_from_w25()) {
        set_system_prompt(g_system_prompt);
    }

    Serial.printf("[Init] PSRAM free: %u KB.\n", (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
    Serial.println(g_model.loaded ? "\nReady. Type a prompt (or /help for commands):"
                                  : "\nNo model loaded. Type /help for commands.");
    Serial.print(">>> ");
}

static char s_input_buffer[kMaxInputChars];
static int s_input_idx = 0;

void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\r' || c == '\n') {
            if (s_input_idx > 0) {
                s_input_buffer[s_input_idx] = '\0';
                Serial.println();
                handle_cli_command(s_input_buffer);
                s_input_idx = 0;
                Serial.print("\n>>> ");
            }
        } else if (c == '\b' || c == 127) {
            if (s_input_idx > 0) {
                s_input_idx--;
                Serial.print("\b \b");
            }
        } else if (s_input_idx < (int)sizeof(s_input_buffer) - 1) {
            s_input_buffer[s_input_idx++] = c;
            Serial.print(c);
        }
    }
}
