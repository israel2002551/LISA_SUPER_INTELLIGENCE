#pragma once
// ==============================================================================
// Core 0: Wi-Fi, telemetry upload, OTA model updates, cloud UART classification
// ==============================================================================
// All server traffic is HTTPS to the evolution server (EDGE_SERVER_BASE_URL),
// authenticated with the node's API key. The device never holds Supabase keys:
// model downloads use short-lived signed URLs the server hands out.
//
//   POST /api/v1/telemetry      ring-log records since the last ack + status
//   GET  /api/v1/ota/check      is a newer model (full image or block delta) available?
//   POST /api/v1/uart/classify  ask the cloud about an unidentified UART stream
// ==============================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <Preferences.h>
#include <ArduinoJson.h>
#include <memory>
#include "edge_config.h"
#include "slot_manager.h"
#include "ring_log.h"
#include "uart_link.h"
#include "inference_task.h"
#include "ota_verify.h"

#define W25_DELTA_MAGIC 0x44353257u   // "W25D" little-endian
#define W25_DELTA_HEADER_BYTES 84

class NetSync {
public:
    void begin(SlotManager* slots, RingLog* ring, UartLink* link, InferenceWorker* infer, Preferences* prefs) {
        slots_ = slots;
        ring_ = ring;
        link_ = link;
        infer_ = infer;
        prefs_ = prefs;
        cursor_ = prefs_->getUInt("ack_next", 1);
    }

    bool configured() const { return strlen(EDGE_WIFI_SSID) > 0 && strlen(EDGE_SERVER_BASE_URL) > 0; }
    uint32_t cursor() const { return cursor_; }
    bool online() const { return WiFi.status() == WL_CONNECTED; }

    void request_ota_check() { force_ota_check_ = true; }

    // One iteration of the net task.
    void step() {
        if (!configured()) {
            vTaskDelay(pdMS_TO_TICKS(5000));
            return;
        }
        if (!ensure_wifi()) {
            vTaskDelay(pdMS_TO_TICKS(1000));
            return;
        }
        uint32_t now = millis();
        bool pending = ring_->next_seq() > cursor_;
        if ((pending && now - last_upload_ms_ >= EDGE_TELEMETRY_INTERVAL_MS) || now - last_upload_ms_ >= EDGE_HEARTBEAT_INTERVAL_MS ||
            last_upload_ms_ == 0) {
            upload_telemetry();
            last_upload_ms_ = millis();
        }
        if (force_ota_check_ || last_ota_ms_ == 0 || now - last_ota_ms_ >= EDGE_OTA_CHECK_INTERVAL_MS) {
            force_ota_check_ = false;
            ota_check();
            last_ota_ms_ = millis();
        }
        classify_raw_device();
        vTaskDelay(pdMS_TO_TICKS(500));
    }

private:
    SlotManager* slots_ = nullptr;
    RingLog* ring_ = nullptr;
    UartLink* link_ = nullptr;
    InferenceWorker* infer_ = nullptr;
    Preferences* prefs_ = nullptr;
    uint32_t cursor_ = 1;   // first ring seq not yet acknowledged by the server
    uint32_t last_upload_ms_ = 0, last_ota_ms_ = 0, last_wifi_attempt_ms_ = 0;
    bool wifi_started_ = false;
    volatile bool force_ota_check_ = false;

    // ---- Wi-Fi ----
    bool ensure_wifi() {
        if (WiFi.status() == WL_CONNECTED) return true;
        if (!wifi_started_ || millis() - last_wifi_attempt_ms_ > EDGE_WIFI_RETRY_MS) {
            if (!wifi_started_) {
                WiFi.mode(WIFI_STA);
                WiFi.setAutoReconnect(true);
                Serial.printf("[Net] Connecting to Wi-Fi '%s'...\n", EDGE_WIFI_SSID);
            }
            WiFi.begin(EDGE_WIFI_SSID, EDGE_WIFI_PASSWORD);
            wifi_started_ = true;
            last_wifi_attempt_ms_ = millis();
        }
        return false;
    }

    // ---- HTTP helpers ----
    static std::unique_ptr<WiFiClient> make_client(const String& url) {
        if (url.startsWith("https://")) {
            std::unique_ptr<WiFiClientSecure> c(new WiFiClientSecure());
#ifdef EDGE_TLS_INSECURE
            c->setInsecure();
#else
            if (strlen(EDGE_TLS_ROOT_CA) == 0) {
                Serial.println("[Net] EDGE_TLS_ROOT_CA is empty; refusing HTTPS without certificate checks.");
                return nullptr;
            }
            c->setCACert(EDGE_TLS_ROOT_CA);
#endif
            return std::unique_ptr<WiFiClient>(c.release());
        }
#ifdef EDGE_ALLOW_PLAIN_HTTP
        if (url.startsWith("http://")) return std::unique_ptr<WiFiClient>(new WiFiClient());
#endif
        Serial.printf("[Net] Refusing URL %s (plain HTTP not allowed).\n", url.c_str());
        return nullptr;
    }

    static String api_url(const char* path) { return String(EDGE_SERVER_BASE_URL) + path; }

    bool request(const char* method, const String& url, const String& body, int& code, String& resp) {
        std::unique_ptr<WiFiClient> client = make_client(url);
        if (!client) return false;
        HTTPClient http;
        http.setTimeout(EDGE_HTTP_TIMEOUT_MS);
        if (!http.begin(*client, url)) return false;
        http.addHeader("Authorization", String("Bearer ") + EDGE_NODE_API_KEY);
        http.addHeader("X-Node-Id", EDGE_NODE_ID);
        if (strcmp(method, "GET") == 0) {
            code = http.GET();
        } else {
            http.addHeader("Content-Type", "application/json");
            code = http.POST(body);
        }
        resp = code > 0 ? http.getString() : http.errorToString(code);
        http.end();
        return code >= 200 && code < 300;
    }

    // ---- Telemetry ----
    String status_json() {
        JsonDocument d;
        d["uptime_s"] = millis() / 1000;
        d["model_loaded"] = g_model.loaded;
        d["model_version"] = g_model.model_version;
        d["selftest_ok"] = infer_->selftest_ok;
        d["inferences"] = infer_->completed;
        JsonObject w = d["w25q"].to<JsonObject>();
        char id[9];
        snprintf(id, sizeof(id), "%06x", (unsigned)g_w25_id);
        w["jedec_id"] = id;
        w["size"] = g_w25_size;
        slots_->describe(d["partitions"].to<JsonObject>());
        JsonObject r = d["ring"].to<JsonObject>();
        r["next_seq"] = ring_->next_seq();
        r["oldest_seq"] = ring_->oldest_seq();
        r["ack_next"] = cursor_;
        r["queued"] = ring_->queued();
        r["dropped"] = ring_->dropped();
        r["lost"] = ring_->oldest_seq() > cursor_ ? ring_->oldest_seq() - cursor_ : 0;
        const LinkStats& s = link_->stats();
        JsonObject l = d["uart"].to<JsonObject>();
        l["state"] = link_state_name(link_->state());
        l["events"] = s.events;
        l["events_dropped"] = s.events_dropped;
        l["actions_sent"] = s.actions_sent;
        l["acks"] = s.acks;
        l["rejects"] = s.rejects;
        l["timeouts"] = s.timeouts;
        l["nops"] = s.nops;
        l["invalid"] = s.invalid;
        d["heap_internal_free"] = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
        d["heap_psram_free"] = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
        d["wifi_rssi"] = WiFi.RSSI();
        String out;
        serializeJson(d, out);
        return out;
    }

    String devices_json() {
        DeviceProfile p;
        link_->snapshot(p);
        if (!p.present) return "[]";
        JsonDocument d;
        JsonObject o = d.add<JsonObject>();
        o["device_key"] = p.key;
        o["port"] = 1;
        o["identified"] = p.identified;
        o["baud"] = p.baud;
        o["link"] = p.link;
        o["format"] = p.format;
        o["type"] = p.type;
        if (p.identified && !p.manifest_truncated) o["manifest"] = serialized(String(p.manifest));
        String out;
        serializeJson(d, out);
        return out;
    }

    void upload_telemetry() {
        String body;
        body.reserve(EDGE_TELEMETRY_BATCH_BYTES + 2048);
        body = "{\"node_id\":\"" EDGE_NODE_ID "\",\"fw_version\":\"" EDGE_FW_VERSION "\",\"status\":";
        body += status_json();
        body += ",\"devices\":";
        body += devices_json();
        body += ",\"events\":[";
        bool first = true;
        uint32_t last_seq = 0;
        ring_->read_from(cursor_, [&](uint32_t seq, uint8_t type, const char* payload, size_t len) {
            if (!first && body.length() + len + 48 > EDGE_TELEMETRY_BATCH_BYTES) return false;
            if (!first) body += ',';
            first = false;
            body += "{\"seq\":";
            body += seq;
            body += ",\"type\":";
            body += type;
            body += ",\"data\":";
            body += payload;
            body += '}';
            last_seq = seq;
            return true;
        });
        body += "]}";

        int code = 0;
        String resp;
        if (!request("POST", api_url("/api/v1/telemetry"), body, code, resp)) {
            Serial.printf("[Net] Telemetry upload failed (%d): %s\n", code, resp.substring(0, 120).c_str());
            return;
        }
        JsonDocument r;
        if (deserializeJson(r, resp) == DeserializationError::Ok) {
            uint32_t acked = r["acked_seq"] | 0u;
            if (acked >= cursor_ && acked < ring_->next_seq()) {
                cursor_ = acked + 1;
                prefs_->putUInt("ack_next", cursor_);
            }
            if (r["rediscover"] | false) link_->request_rediscover();
        }
        prefs_->putUInt("seq_hw", ring_->next_seq());
        if (last_seq) Serial.printf("[Net] Uploaded events up to seq %u.\n", (unsigned)last_seq);
    }

    // ---- Cloud classification of unidentified devices ----
    void classify_raw_device() {
        ClassifyRequest q;
        if (!link_->take_classify_request(q)) return;
        JsonDocument d;
        d["node_id"] = EDGE_NODE_ID;
        d["device_key"] = q.device_key;
        d["baud"] = q.baud;
        d["format_guess"] = q.format;
        d["sample_hex"] = q.sample_hex;
        String body, resp;
        serializeJson(d, body);
        int code = 0;
        if (!request("POST", api_url("/api/v1/uart/classify"), body, code, resp)) {
            Serial.printf("[Net] Classify request failed (%d).\n", code);
            return;
        }
        JsonDocument r;
        if (deserializeJson(r, resp) == DeserializationError::Ok) {
            link_->apply_classification(r["device_type"] | "", r["format"] | "");
            Serial.printf("[Net] Cloud classified %s as '%s' (%s).\n", q.device_key, (const char*)(r["device_type"] | "?"),
                          (const char*)(r["format"] | "?"));
        }
    }

    // ---- OTA ----
    void log_ota(const char* stage, uint32_t version, const char* detail) {
        JsonDocument d;
        d["kind"] = "ota";
        d["stage"] = stage;
        d["version"] = version;
        d["detail"] = detail;
        char buf[256];
        serializeJson(d, buf, sizeof(buf));
        ring_->post(REC_OTA, buf);
        Serial.printf("[OTA] %s v%u %s\n", stage, (unsigned)version, detail);
    }

    void ota_check() {
        char sha[65];
        slots_->active_sha_hex(sha);
        const SlotInfo& a = slots_->info(slots_->active());
        String url = api_url("/api/v1/ota/check") + "?node_id=" EDGE_NODE_ID "&current_version=" + String(a.version) +
                     "&current_sha256=" + sha;
        int code = 0;
        String resp;
        if (!request("GET", url, "", code, resp)) {
            Serial.printf("[Net] OTA check failed (%d).\n", code);
            return;
        }
        JsonDocument m;
        if (deserializeJson(m, resp) != DeserializationError::Ok) return;
        if (m["rollback"] | false) {
            log_ota("rollback_requested", a.version, "server asked to reject the active model");
            if (slots_->rollback()) reboot();
            return;
        }
        if (m["update"] | false) ota_apply(m);
    }

    bool ota_apply(JsonDocument& m) {
        uint32_t version = m["version"] | 0u;
        uint32_t size = m["size"] | 0u;
        const char* sha_hex = m["sha256"] | "";
        const char* kind = m["kind"] | "full";
        const char* url = m["url"] | "";
        uint8_t sha[32];
        if (size < W25_MODEL_HEADER_BYTES || size > W25_SLOT_BYTES || !sha256_from_hex(sha_hex, sha) || !*url) {
            log_ota("rejected", version, "malformed manifest");
            return false;
        }
        char msg[128];
        ota_manifest_string(version, size, sha_hex, msg, sizeof(msg));
        if (!ota_verify_signature(msg, m["signature"] | "")) {
            log_ota("rejected", version, "bad signature");
            return false;
        }
        if (!slots_->try_lock_staging()) return false;

        log_ota("download", version, kind);
        bool ok = slots_->erase_staging(size);
        if (!ok) log_ota("failed", version, "erase failed");

        std::unique_ptr<WiFiClient> client = ok ? make_client(url) : nullptr;
        HTTPClient http;
        if (ok) {
            ok = client && http.begin(*client, url);
            http.setTimeout(EDGE_HTTP_TIMEOUT_MS);
            int code = ok ? http.GET() : -1;
            int len = code == 200 ? http.getSize() : -1;
            if (code != 200 || len <= 0) {
                // len <= 0 means chunked or unknown length; storage serves Content-Length.
                ok = false;
                snprintf(msg, sizeof(msg), "download HTTP %d, length %d", code, len);
                log_ota("failed", version, msg);
            }
        }
        if (ok) {
            WiFiClient* s = http.getStreamPtr();
            ok = strcmp(kind, "delta") == 0 ? apply_delta(http, s, size, sha) : copy_full(http, s, size);
            if (!ok) log_ota("failed", version, "transfer or patch failed");
        }
        http.end();

        if (ok && !slots_->finalize_staging(version, size, sha, false)) {
            ok = false;
            log_ota("failed", version, "SHA-256 mismatch after write");
        }
        slots_->unlock_staging();
        if (!ok) return false;

        log_ota("switched", version, "rebooting into the new model (trial)");
        reboot();
        return true;
    }

    static bool read_exact(HTTPClient& http, WiFiClient* s, uint8_t* buf, size_t n) {
        size_t got = 0;
        uint32_t last = millis();
        while (got < n) {
            int avail = s->available();
            if (avail > 0) {
                int r = s->read(buf + got, min((size_t)avail, n - got));
                if (r > 0) {
                    got += r;
                    last = millis();
                }
            } else {
                if (!http.connected() || millis() - last > EDGE_HTTP_TIMEOUT_MS) return false;
                delay(2);
            }
        }
        return true;
    }

    bool copy_full(HTTPClient& http, WiFiClient* s, uint32_t size) {
        if ((uint32_t)http.getSize() != size) return false;
        static uint8_t buf[4096];
        for (uint32_t off = 0; off < size; off += sizeof(buf)) {
            uint32_t n = min((uint32_t)sizeof(buf), size - off);
            if (!read_exact(http, s, buf, n) || !slots_->write_staging(off, buf, n)) return false;
        }
        return true;
    }

    // Block delta (server/app/evolution/delta.py): unchanged 4 KiB blocks are copied
    // from the active slot, changed ones come from the stream.
    bool apply_delta(HTTPClient& http, WiFiClient* s, uint32_t size, const uint8_t target_sha[32]) {
        uint8_t hdr[W25_DELTA_HEADER_BYTES];
        if (!read_exact(http, s, hdr, sizeof(hdr))) return false;
        uint32_t magic, fmt, block, target_size, n_blocks;
        memcpy(&magic, hdr, 4);
        memcpy(&fmt, hdr + 4, 4);
        memcpy(&block, hdr + 8, 4);
        memcpy(&target_size, hdr + 12, 4);
        const uint8_t* base_sha = hdr + 16;
        const uint8_t* tgt_sha = hdr + 48;
        memcpy(&n_blocks, hdr + 80, 4);
        const SlotInfo& active = slots_->info(slots_->active());
        if (magic != W25_DELTA_MAGIC || fmt != 1 || block != 4096 || target_size != size ||
            n_blocks != (size + block - 1) / block || memcmp(tgt_sha, target_sha, 32) != 0 ||
            memcmp(base_sha, active.sha256, 32) != 0) {
            Serial.println("[OTA] Delta header doesn't match this device's active model.");
            return false;
        }
        static uint8_t bitmap[(W25_SLOT_BYTES / 4096 + 7) / 8];
        uint32_t bitmap_len = (n_blocks + 7) / 8;
        if (bitmap_len > sizeof(bitmap) || !read_exact(http, s, bitmap, bitmap_len)) return false;

        static uint8_t buf[4096];
        int src = slots_->active();
        for (uint32_t i = 0; i < n_blocks; i++) {
            uint32_t off = i * block;
            uint32_t n = min(block, size - off);
            bool changed = bitmap[i / 8] & (1u << (i % 8));
            bool ok = changed ? read_exact(http, s, buf, n) : slots_->read_slot(src, off, buf, n);
            if (!ok || !slots_->write_staging(off, buf, n)) return false;
        }
        return true;
    }

    void reboot() {
        ring_->flush(2000);
        Serial.println("[Net] Rebooting...");
        delay(200);
        ESP.restart();
    }
};
