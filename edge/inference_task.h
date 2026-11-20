#pragma once
// ==============================================================================
// Core 1: local Gemma inference
// ==============================================================================
// The only task that touches the engine. Takes prompts from the UART task,
// runs greedy decoding on the model copied into PSRAM at boot, and returns the
// first line of output. It never touches the UART or flash, so a slow
// generation can't stall hardware I/O.
// ==============================================================================

#include "gemma4_engine.h"
#include "edge_config.h"
#include "edge_messages.h"

static_assert(EDGE_PROMPT_MAX_TOKENS + EDGE_ACTION_MAX_TOKENS <= GEMMA_MAX_CONTEXT_LENGTH,
              "prompt + action token budgets must fit the model context");

class InferenceWorker {
public:
    volatile bool selftest_ok = false;
    volatile bool selftest_done = false;
    volatile uint32_t completed = 0;

    // Runs once on the inference task before it takes requests. Checks that the
    // loaded weights produce finite logits and decodable tokens.
    void self_test() {
        if (!g_model.loaded) {
            selftest_done = true;
            return;
        }
        InferResult r;
        InferRequest q{};
        strlcpy(q.prompt, "<d>selftest:none<e>{}<a>", sizeof(q.prompt));
        q.queued_ms = millis();
        run(q, r);
        bool finite = true;
        for (int i = 0; i < GEMMA_VOCAB_SIZE; i += 97) finite &= isfinite(g_logits[i]);
        selftest_ok = finite;
        selftest_done = true;
        Serial.printf("[Infer] Self-test %s in %u ms: \"%s\"\n", finite ? "passed" : "FAILED (non-finite logits)",
                      (unsigned)r.latency_ms, r.output);
    }

    void run(const InferRequest& q, InferResult& r) {
        r.act_id = q.act_id;
        r.model_version = g_model.model_version;
        r.model_ready = g_model.loaded;
        r.output[0] = '\0';
        if (g_model.loaded) {
            // Tokenize in full, then cut to the token budget (same rule as the server).
            int n = gemma_tokenize(q.prompt, &tokens_[1], kMaxPromptChars);
            n = min(n, EDGE_PROMPT_MAX_TOKENS - 1);
            tokens_[0] = GEMMA_BOS_TOKEN_ID;
            int n_out = gemma_generate_greedy(tokens_, n + 1, out_tokens_, EDGE_ACTION_MAX_TOKENS);

            size_t len = 0;
            for (int i = 0; i < n_out; i++) {
                String piece = gemma_decode_token(out_tokens_[i]);
                int nl = piece.indexOf('\n');
                if (nl >= 0) piece = piece.substring(0, nl);
                if (len + piece.length() >= sizeof(r.output)) break;
                memcpy(&r.output[len], piece.c_str(), piece.length());
                len += piece.length();
                if (nl >= 0) break;
            }
            r.output[len] = '\0';
        }
        r.latency_ms = millis() - q.queued_ms;
        completed++;
    }

private:
    static const int kMaxPromptChars = 200;
    uint32_t tokens_[kMaxPromptChars + 1];
    uint32_t out_tokens_[EDGE_ACTION_MAX_TOKENS];
};
