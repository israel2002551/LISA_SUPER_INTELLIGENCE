#pragma once
// ==============================================================================
// OTA manifest signatures (ECDSA P-256 / SHA-256)
// ==============================================================================
// The server signs   "W25OTA1|<version>|<image size>|<image sha256 hex>"
// with its private key (server/app/evolution/signing.py). The device checks the
// signature with EDGE_OTA_PUBLIC_KEY_PEM before downloading anything, then checks
// the downloaded image against the signed SHA-256. A compromised download URL,
// storage bucket or TLS link can't install weights the server didn't sign.
// ==============================================================================

#include <Arduino.h>
#include "mbedtls/pk.h"
#include "mbedtls/base64.h"
#include "mbedtls/sha256.h"
#include "edge_config.h"

static void ota_manifest_string(uint32_t version, uint32_t size, const char* sha256_hex, char* out, size_t out_len) {
    snprintf(out, out_len, "W25OTA1|%u|%u|%s", (unsigned)version, (unsigned)size, sha256_hex);
}

// signature_b64: base64 of the DER-encoded ECDSA signature.
static bool ota_verify_signature(const char* message, const char* signature_b64) {
#ifdef EDGE_ALLOW_UNSIGNED_OTA
    Serial.println("[OTA] WARNING: EDGE_ALLOW_UNSIGNED_OTA is set; signature not checked.");
    return true;
#else
    if (strlen(EDGE_OTA_PUBLIC_KEY_PEM) == 0) {
        Serial.println("[OTA] No EDGE_OTA_PUBLIC_KEY_PEM in secrets.h; refusing unsigned updates.");
        return false;
    }
    if (!signature_b64 || !*signature_b64) return false;

    uint8_t sig[80];
    size_t sig_len = 0;
    if (mbedtls_base64_decode(sig, sizeof(sig), &sig_len, (const unsigned char*)signature_b64, strlen(signature_b64)) != 0) {
        return false;
    }
    uint8_t hash[32];
    if (mbedtls_sha256_ret((const unsigned char*)message, strlen(message), hash, 0) != 0) return false;

    mbedtls_pk_context pk;
    mbedtls_pk_init(&pk);
    int rc = mbedtls_pk_parse_public_key(&pk, (const unsigned char*)EDGE_OTA_PUBLIC_KEY_PEM, strlen(EDGE_OTA_PUBLIC_KEY_PEM) + 1);
    if (rc == 0) rc = mbedtls_pk_verify(&pk, MBEDTLS_MD_SHA256, hash, sizeof(hash), sig, sig_len);
    mbedtls_pk_free(&pk);
    return rc == 0;
#endif
}
