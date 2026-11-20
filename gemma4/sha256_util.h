#pragma once
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "mbedtls/sha256.h"

// Streaming SHA-256 (mbedtls 2.28 "_ret" API, hardware-accelerated on the ESP32-S3).
struct Sha256 {
    mbedtls_sha256_context ctx;

    Sha256() {
        mbedtls_sha256_init(&ctx);
        mbedtls_sha256_starts_ret(&ctx, 0);
    }
    ~Sha256() { mbedtls_sha256_free(&ctx); }

    void update(const void* data, size_t len) {
        mbedtls_sha256_update_ret(&ctx, (const unsigned char*)data, len);
    }
    void finish(uint8_t out[32]) { mbedtls_sha256_finish_ret(&ctx, out); }
};

static inline void sha256_to_hex(const uint8_t digest[32], char out[65]) {
    for (int i = 0; i < 32; i++) snprintf(&out[i * 2], 3, "%02x", digest[i]);
    out[64] = '\0';
}

static inline bool sha256_from_hex(const char* hex, uint8_t out[32]) {
    if (!hex || strlen(hex) != 64) return false;
    for (int i = 0; i < 32; i++) {
        unsigned v;
        if (sscanf(&hex[i * 2], "%2x", &v) != 1) return false;
        out[i] = (uint8_t)v;
    }
    return true;
}
