#pragma once
#include <Arduino.h>
#include <stdint.h>
#include <string.h>

// Vocabulary size and special tokens. By default they come from cluster_config.h
// (llama2.c's 32,000-token tokenizer.bin). A build can instead define
// BPE_VOCAB_SIZE, BPE_BOS_TOKEN_ID and BPE_EOS_TOKEN_ID before including this
// file; only the first BPE_VOCAB_SIZE entries of tokenizer.bin are then loaded.
#ifndef BPE_VOCAB_SIZE
#include "cluster_config.h"
#define BPE_VOCAB_SIZE   VOCAB_SIZE
#define BPE_BOS_TOKEN_ID BOS_TOKEN_ID
#define BPE_EOS_TOKEN_ID EOS_TOKEN_ID
#endif
#if !defined(BPE_BOS_TOKEN_ID) || !defined(BPE_EOS_TOKEN_ID)
#error "Define BPE_BOS_TOKEN_ID and BPE_EOS_TOKEN_ID together with BPE_VOCAB_SIZE"
#endif

// ==============================================================================
// Lightweight BPE Tokenizer for ESP32-S3
// Compatible with Karpathy's llama2.c tokenizer.bin (32,000 vocab)
// ==============================================================================

class BpeTokenizer {
public:
    int32_t max_token_length;
    const char* vocab[BPE_VOCAB_SIZE];
    uint8_t vocab_len[BPE_VOCAB_SIZE];
    float vocab_score[BPE_VOCAB_SIZE];
    int16_t byte_to_token[256];
    bool initialized;

    BpeTokenizer() : max_token_length(0), initialized(false) {
        for (int i = 0; i < 256; i++) byte_to_token[i] = -1;
    }

    bool init(const uint8_t* data, size_t total_size) {
        if (!data || total_size < 4) return false;
        
        memcpy(&max_token_length, data, sizeof(int32_t));
        const uint8_t* ptr = data + 4;
        const uint8_t* end = data + total_size;

        for (int i = 0; i < BPE_VOCAB_SIZE; i++) {
            if (ptr + 8 > end) return false;
            memcpy(&vocab_score[i], ptr, sizeof(float));
            ptr += 4;
            int32_t len;
            memcpy(&len, ptr, sizeof(int32_t));
            ptr += 4;
            if (ptr + len > end) return false;
            vocab[i] = (const char*)ptr;
            vocab_len[i] = (uint8_t)len;
            
            // Map individual 1-byte characters for O(1) initial lookup
            if (len == 1) {
                uint8_t byte_val = (uint8_t)vocab[i][0];
                byte_to_token[byte_val] = i;
            }
            ptr += len;
        }

        initialized = true;
        return true;
    }

    int find_token(const char* s, int len) const {
        if (len == 1) {
            int16_t tid = byte_to_token[(uint8_t)s[0]];
            if (tid != -1) return tid;
        }
        for (int i = 0; i < BPE_VOCAB_SIZE; i++) {
            if (vocab_len[i] == len && memcmp(vocab[i], s, len) == 0) {
                return i;
            }
        }
        return -1;
    }

    int encode(const String& text, uint32_t* out_tokens, int max_tokens) const {
        if (!initialized || text.length() == 0) return 0;
        
        // 1. Initial byte encoding
        int num_tokens = 0;
        for (unsigned int i = 0; i < text.length() && num_tokens < max_tokens; i++) {
            char c = text[i];
            int tid = find_token(&c, 1);
            if (tid != -1) {
                out_tokens[num_tokens++] = (uint32_t)tid;
            } else {
                // Byte fallback e.g. <0xNN>
                char byte_str[7];
                snprintf(byte_str, sizeof(byte_str), "<0x%02X>", (uint8_t)c);
                int b_tid = find_token(byte_str, 6);
                out_tokens[num_tokens++] = (b_tid != -1) ? (uint32_t)b_tid : 0;
            }
        }

        // 2. Iterative BPE Merge
        char merge_buf[64];
        while (num_tokens >= 2) {
            float best_score = -1e9f;
            int best_idx = -1;
            int best_token = -1;

            for (int i = 0; i < num_tokens - 1; i++) {
                int len1 = vocab_len[out_tokens[i]];
                int len2 = vocab_len[out_tokens[i + 1]];
                if (len1 + len2 >= (int)sizeof(merge_buf)) continue;

                memcpy(merge_buf, vocab[out_tokens[i]], len1);
                memcpy(merge_buf + len1, vocab[out_tokens[i + 1]], len2);
                int merged_len = len1 + len2;

                int tid = find_token(merge_buf, merged_len);
                if (tid != -1 && vocab_score[tid] > best_score) {
                    best_score = vocab_score[tid];
                    best_idx = i;
                    best_token = tid;
                }
            }

            if (best_idx == -1) break; // No more pairs to merge

            out_tokens[best_idx] = (uint32_t)best_token;
            for (int i = best_idx + 1; i < num_tokens - 1; i++) {
                out_tokens[i] = out_tokens[i + 1];
            }
            num_tokens--;
        }

        return num_tokens;
    }

    String decode(uint32_t token_id) const {
        if (!initialized || token_id >= BPE_VOCAB_SIZE) {
            char buf[16];
            snprintf(buf, sizeof(buf), "[%u]", token_id);
            return String(buf);
        }
        if (token_id == BPE_EOS_TOKEN_ID || token_id == BPE_BOS_TOKEN_ID) return "";

        // Check for byte tokens like <0x0A>
        if (vocab_len[token_id] == 6 && strncmp(vocab[token_id], "<0x", 3) == 0) {
            char hex[3] = {vocab[token_id][3], vocab[token_id][4], '\0'};
            char c = (char)strtol(hex, NULL, 16);
            return String(c);
        }

        // Regular token string
        char buf[64];
        int len = min((int)vocab_len[token_id], 63);
        memcpy(buf, vocab[token_id], len);
        buf[len] = '\0';
        return String(buf);
    }
};
