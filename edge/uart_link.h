#pragma once
// ==============================================================================
// UART auto-discovery, handshake, bus upgrade and action loop (Core 0)
// ==============================================================================
// State machine per port:
//
//   DISCOVER ──IDENTIFY answered──► IDENTIFIED ──(UPGRADE + PING/PONG)──► uart_fast
//      │                               │  events -> inference -> validated action
//      │                               │  -> {"cmd":..,"id":N} -> {"ack":N,"ok":..}
//      ├──bytes but no answer──────► RAW (format analyzer, samples logged,
//      │                                  cloud classification requested)
//      └──silence on every baud────► NO_DEVICE (retry later)
//
// Protocol (docs/ARCHITECTURE.md, tools/fake_peripheral.py implements the device side):
//   host -> {"cmd":"IDENTIFY","host":"edge-01","ai":"LISA","proto":1}
//   dev  -> {"id":"thermo-01","type":"thermostat","accepts":[...],"bus":{"max_baud":2000000,"links":["uart"]}}
//   host -> {"cmd":"UPGRADE","link":"uart","baud":2000000}   dev -> {"ack":"UPGRADE","ok":true}
//   both switch baud; host -> {"cmd":"PING","n":123}          dev -> {"pong":123}
//   (the device reverts to the old baud if no PING arrives within 1 s)
//   dev  -> {"ev":"temp","v":31.2}            event, fed to the model
//   host -> {"cmd":"set_fan","args":{"level":2},"id":42}
//   dev  -> {"ack":42,"ok":true}              action result
//   dev  -> {"fb":"good","id":42}             optional feedback (reward signal)
// ==============================================================================

#include <Arduino.h>
#include <ArduinoJson.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "edge_config.h"
#include "edge_messages.h"
#include "action_schema.h"
#include "ring_log.h"

enum LinkState { LINK_DISCOVER, LINK_IDENTIFIED, LINK_RAW, LINK_NO_DEVICE };

static const char* link_state_name(LinkState s) {
    switch (s) {
        case LINK_DISCOVER: return "discover";
        case LINK_IDENTIFIED: return "identified";
        case LINK_RAW: return "raw";
        default: return "no_device";
    }
}

// ---- Raw stream analysis ----------------------------------------------------------
static uint16_t modbus_crc16(const uint8_t* d, size_t n) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < n; i++) {
        crc ^= d[i];
        for (int b = 0; b < 8; b++) crc = (crc & 1) ? (crc >> 1) ^ 0xA001 : crc >> 1;
    }
    return crc;
}

static bool is_text_byte(uint8_t c) { return (c >= 0x20 && c < 0x7F) || c == '\r' || c == '\n' || c == '\t'; }

// Guesses the wire format of an unidentified device from its first bytes.
// Mirrored by server/app/uart_analysis.py (which can also ask Claude).
static void analyze_raw(const uint8_t* buf, size_t n, char* fmt, size_t fmt_len) {
    if (n == 0) {
        strlcpy(fmt, "unknown", fmt_len);
        return;
    }
    size_t text = 0, lines = 0, json = 0, nmea = 0, csv = 0;
    for (size_t i = 0; i < n; i++) text += is_text_byte(buf[i]);
    if (text * 10 >= n * 9) {
        // Classify line by line
        size_t start = 0;
        for (size_t i = 0; i <= n; i++) {
            if (i == n || buf[i] == '\n') {
                size_t a = start, b = i;
                while (a < b && (buf[a] == ' ' || buf[a] == '\r')) a++;
                while (b > a && (buf[b - 1] == ' ' || buf[b - 1] == '\r')) b--;
                if (b > a) {
                    lines++;
                    size_t commas = 0;
                    bool star = false;
                    for (size_t k = a; k < b; k++) {
                        commas += buf[k] == ',';
                        star |= buf[k] == '*';
                    }
                    if (buf[a] == '{' && buf[b - 1] == '}') json++;
                    else if (buf[a] == '$' && star) nmea++;
                    else if (commas >= 2) csv++;
                }
                start = i + 1;
            }
        }
        const char* f = "ascii_lines";
        if (lines && json * 2 >= lines) f = "json_lines";
        else if (lines && nmea * 2 >= lines) f = "nmea";
        else if (lines && csv * 2 >= lines) f = "csv";
        strlcpy(fmt, f, fmt_len);
        return;
    }
    // Binary: look for a Modbus RTU frame (CRC-16 over the leading bytes)
    for (size_t len = 4; len <= n && len <= 256; len++) {
        uint16_t crc = modbus_crc16(buf, len - 2);
        if ((crc & 0xFF) == buf[len - 2] && (crc >> 8) == buf[len - 1]) {
            strlcpy(fmt, "modbus_rtu", fmt_len);
            return;
        }
    }
    strlcpy(fmt, "binary", fmt_len);
}

// Passive-listen score for one baud rate: real data at the right baud has few
// framing errors and is either text or 0x55 sync bytes.
static float score_sample(const uint8_t* buf, size_t n, uint32_t frame_errors) {
    if (n == 0) return 0.0f;
    size_t text = 0, sync = 0;
    for (size_t i = 0; i < n; i++) {
        text += is_text_byte(buf[i]);
        sync += buf[i] == 0x55;
    }
    float printable = (float)text / n, sync_ratio = (float)sync / n;
    float err_ratio = (float)frame_errors / n;
    float penalty = 1.0f - fminf(1.0f, 4.0f * err_ratio);
    return penalty * (0.5f + 0.5f * fmaxf(printable, sync_ratio));
}

struct ClassifyRequest {
    char device_key[40];
    uint32_t baud;
    char format[16];
    char sample_hex[257];
};

struct LinkStats {
    uint32_t events = 0, events_dropped = 0, actions_sent = 0, acks = 0, rejects = 0, timeouts = 0;
    uint32_t nops = 0, invalid = 0, discoveries = 0;
};

class UartLink {
public:
    UartLink(HardwareSerial& port, uint8_t port_no, int rx_pin, int tx_pin)
        : port_(port), port_no_(port_no), rx_pin_(rx_pin), tx_pin_(tx_pin) {}

    void begin(RingLog* ring, QueueHandle_t infer_q, QueueHandle_t result_q, uint32_t boot_count) {
        ring_ = ring;
        infer_q_ = infer_q;
        result_q_ = result_q;
        act_seq_ = (boot_count & 0x7FF) << 20;   // act ids unique across reboots
        mutex_ = xSemaphoreCreateMutex();
        port_.setRxBufferSize(EDGE_UART_RX_BUFFER);
        port_.onReceiveError([this](hardwareSerial_error_t e) {
            if (e == UART_FRAME_ERROR || e == UART_PARITY_ERROR || e == UART_BREAK_ERROR) frame_errors_++;
        });
        port_.begin(kEdgeBaudCandidates[0], SERIAL_8N1, rx_pin_, tx_pin_);
    }

    // One iteration of the UART task.
    void step() {
        if (rediscover_ || state_ == LINK_DISCOVER) {
            discover();
            return;
        }
        if (state_ == LINK_NO_DEVICE) {
            if (millis() - last_attempt_ms_ > EDGE_NO_DEVICE_RETRY_MS) state_ = LINK_DISCOVER;
            vTaskDelay(pdMS_TO_TICKS(20));
            return;
        }
        pump_rx();
        pump_results();
        expire_pending();
        keepalive();
        vTaskDelay(1);
    }

    void request_rediscover() { rediscover_ = true; }
    LinkState state() const { return state_; }
    const LinkStats& stats() const { return stats_; }

    void snapshot(DeviceProfile& out) {
        xSemaphoreTake(mutex_, portMAX_DELAY);
        out = profile_;
        xSemaphoreGive(mutex_);
    }

    // Net task: fetch a pending cloud-classification request for a raw device.
    bool take_classify_request(ClassifyRequest& out) {
        xSemaphoreTake(mutex_, portMAX_DELAY);
        bool has = classify_pending_;
        if (has) {
            out = classify_req_;
            classify_pending_ = false;
        }
        xSemaphoreGive(mutex_);
        return has;
    }

    // Net task: the cloud's answer for a raw device.
    void apply_classification(const char* device_type, const char* format) {
        xSemaphoreTake(mutex_, portMAX_DELAY);
        if (!profile_.identified) {
            if (device_type && *device_type) strlcpy(profile_.type, device_type, sizeof(profile_.type));
            if (format && *format) strlcpy(profile_.format, format, sizeof(profile_.format));
        }
        xSemaphoreGive(mutex_);
    }

private:
    struct InFlight {
        bool used = false;
        bool awaiting_ack = false;
        uint32_t act_id = 0;
        uint32_t deadline_ms = 0;
        uint32_t latency_ms = 0;
        uint32_t model_version = 0;
        char input[EDGE_EVENT_MAX_CHARS + 1];
        char prompt[200];
        char output[160];
        char action[160];
    };
    static const int kMaxInFlight = 8;

    HardwareSerial& port_;
    uint8_t port_no_;
    int rx_pin_, tx_pin_;
    RingLog* ring_ = nullptr;
    QueueHandle_t infer_q_ = nullptr, result_q_ = nullptr;
    SemaphoreHandle_t mutex_ = nullptr;

    LinkState state_ = LINK_DISCOVER;
    DeviceProfile profile_;
    LinkStats stats_;
    volatile uint32_t frame_errors_ = 0;
    volatile bool rediscover_ = false;
    uint32_t act_seq_ = 0;
    uint32_t last_rx_ms_ = 0, last_attempt_ms_ = 0, last_observation_ms_ = 0;
    uint32_t ping_nonce_ = 0, ping_deadline_ms_ = 0;
    uint8_t missed_pings_ = 0;
    bool classify_pending_ = false;
    ClassifyRequest classify_req_{};

    char line_[512];
    size_t line_len_ = 0;
    DeviceProfile scratch_;          // working copy during discovery / re-identification
    char scratch_line_[512];
    uint8_t sample_[256];
    uint8_t best_sample_[256];
    uint8_t raw_[256];
    size_t raw_len_ = 0;
    uint32_t raw_last_byte_ms_ = 0;
    InFlight inflight_[kMaxInFlight];

    // ---- Low-level port helpers ----
    void set_baud(uint32_t baud) {
        port_.flush();
        port_.updateBaudRate(baud);
        delay(5);
        drain();
    }
    void drain() {
        while (port_.available()) port_.read();
        line_len_ = 0;
    }
    void send_line(const char* s) {
        port_.write((const uint8_t*)s, strlen(s));
        port_.write('\n');
    }
    // Reads one '\n'-terminated line within timeout_ms (without the newline).
    bool read_line(char* out, size_t cap, uint32_t timeout_ms) {
        uint32_t t0 = millis();
        size_t n = 0;
        while (millis() - t0 < timeout_ms) {
            while (port_.available()) {
                char c = port_.read();
                if (c == '\n') {
                    while (n && out[n - 1] == '\r') n--;
                    out[n] = '\0';
                    return true;
                }
                if (n < cap - 1) out[n++] = c;
            }
            vTaskDelay(1);
        }
        return false;
    }
    size_t listen(uint32_t ms, uint8_t* buf, size_t cap) {
        uint32_t t0 = millis();
        size_t n = 0;
        while (millis() - t0 < ms) {
            while (port_.available() && n < cap) buf[n++] = port_.read();
            vTaskDelay(1);
        }
        return n;
    }

    uint32_t next_act_id() { return ++act_seq_; }

    // ---- Discovery ----
    void discover() {
        rediscover_ = false;
        stats_.discoveries++;
        abort_inflight("link reset");
        state_ = LINK_DISCOVER;
        // Big buffers are members, not locals: the UART task's stack is small.
        DeviceProfile& p = scratch_;
        char* line = scratch_line_;

        // 1. Active handshake: IDENTIFY at each candidate baud rate
        char ident[128];
        snprintf(ident, sizeof(ident), "{\"cmd\":\"IDENTIFY\",\"host\":\"%s\",\"ai\":\"%s\",\"proto\":1}", EDGE_NODE_ID,
                 EDGE_AI_NAME);
        for (uint32_t baud : kEdgeBaudCandidates) {
            set_baud(baud);
            send_line(ident);
            uint32_t t0 = millis();
            while (millis() - t0 < EDGE_HANDSHAKE_TIMEOUT_MS) {
                if (read_line(line, sizeof(scratch_line_), EDGE_HANDSHAKE_TIMEOUT_MS - (millis() - t0)) &&
                    profile_from_manifest(line, baud, p)) {
                    on_identified(p);
                    return;
                }
            }
        }

        // 2. Passive: listen at each baud and score what arrives
        float best = 0.0f;
        uint32_t best_baud = 0;
        size_t best_n = 0;
        for (uint32_t baud : kEdgeBaudCandidates) {
            set_baud(baud);
            frame_errors_ = 0;
            size_t n = listen(EDGE_PASSIVE_LISTEN_MS, sample_, sizeof(sample_));
            float s = score_sample(sample_, n, frame_errors_);
            if (s > best) {
                best = s;
                best_baud = baud;
                memcpy(best_sample_, sample_, n);
                best_n = n;
            }
        }
        if (best > 0.4f) {
            on_raw(best_baud, best_sample_, best_n);
            return;
        }

        if (state_ != LINK_NO_DEVICE) Serial.printf("[UART%u] No device found; retrying every %u s.\n", port_no_, EDGE_NO_DEVICE_RETRY_MS / 1000);
        xSemaphoreTake(mutex_, portMAX_DELAY);
        profile_ = empty_profile();
        xSemaphoreGive(mutex_);
        state_ = LINK_NO_DEVICE;
        last_attempt_ms_ = millis();
    }

    void on_identified(DeviceProfile& p) {
        uint32_t from = p.baud;
        const char* upgrade = try_upgrade(p);
        xSemaphoreTake(mutex_, portMAX_DELAY);
        profile_ = p;
        xSemaphoreGive(mutex_);
        state_ = LINK_IDENTIFIED;
        last_rx_ms_ = millis();
        missed_pings_ = 0;
        ping_deadline_ms_ = 0;
        Serial.printf("[UART%u] Identified '%s' (%s) at %u baud, link %s, %u commands.\n", port_no_, p.key, p.type,
                      (unsigned)p.baud, p.link, p.n_accepts);

        JsonDocument d;
        d["kind"] = "handshake";
        d["dev"] = p.key;
        d["result"] = upgrade;
        d["from_baud"] = from;
        d["baud"] = p.baud;
        d["link"] = p.link;
        log_doc(REC_HANDSHAKE, d);
        log_device(p, nullptr, 0);
    }

    void on_raw(uint32_t baud, const uint8_t* sample, size_t n) {
        DeviceProfile& p = scratch_;
        p = empty_profile();
        p.present = true;
        p.identified = false;
        p.baud = baud;
        snprintf(p.key, sizeof(p.key), "raw:uart%u", port_no_);
        strlcpy(p.type, "unknown", sizeof(p.type));
        analyze_raw(sample, n, p.format, sizeof(p.format));
        set_baud(baud);
        xSemaphoreTake(mutex_, portMAX_DELAY);
        profile_ = p;
        classify_pending_ = true;
        strlcpy(classify_req_.device_key, p.key, sizeof(classify_req_.device_key));
        classify_req_.baud = baud;
        strlcpy(classify_req_.format, p.format, sizeof(classify_req_.format));
        size_t hn = min(n, (size_t)128);
        for (size_t i = 0; i < hn; i++) snprintf(&classify_req_.sample_hex[i * 2], 3, "%02x", sample[i]);
        classify_req_.sample_hex[hn * 2] = '\0';
        xSemaphoreGive(mutex_);
        state_ = LINK_RAW;
        last_rx_ms_ = millis();
        Serial.printf("[UART%u] Unidentified device at %u baud, format guess '%s'.\n", port_no_, (unsigned)baud, p.format);
        log_device(p, sample, n);
    }

    // ---- Bus upgrade ----
    // Returns a short result string for the handshake log.
    const char* try_upgrade(DeviceProfile& p) {
        bool uart_ok = p.links[0] == '\0' || strstr(p.links, "uart") != nullptr;
        uint32_t target = min(p.max_baud, (uint32_t)EDGE_UART_UPGRADE_MAX_BAUD);
        if (uart_ok && target > p.baud) {
            char msg[96], line[160];
            snprintf(msg, sizeof(msg), "{\"cmd\":\"UPGRADE\",\"link\":\"uart\",\"baud\":%u}", (unsigned)target);
            send_line(msg);
            bool acked = false;
            uint32_t t0 = millis();
            while (!acked && millis() - t0 < EDGE_UPGRADE_TIMEOUT_MS) {
                if (!read_line(line, sizeof(line), EDGE_UPGRADE_TIMEOUT_MS)) break;
                JsonDocument d;
                if (deserializeJson(d, line) == DeserializationError::Ok && strcmp(d["ack"] | "", "UPGRADE") == 0) {
                    acked = d["ok"] | false;
                    break;
                }
            }
            if (!acked) return "upgrade_declined";

            uint32_t old = p.baud;
            set_baud(target);
            delay(20);
            uint32_t nonce = esp_random() & 0xFFFF;
            snprintf(msg, sizeof(msg), "{\"cmd\":\"PING\",\"n\":%u}", (unsigned)nonce);
            send_line(msg);
            t0 = millis();
            while (millis() - t0 < EDGE_UPGRADE_TIMEOUT_MS) {
                if (!read_line(line, sizeof(line), EDGE_UPGRADE_TIMEOUT_MS)) break;
                JsonDocument d;
                if (deserializeJson(d, line) == DeserializationError::Ok && (d["pong"] | 0xFFFFFFFFu) == nonce) {
                    p.baud = target;
                    strlcpy(p.link, "uart_fast", sizeof(p.link));
                    return "upgrade_ok";
                }
            }
            set_baud(old);   // the device falls back on its own after 1 s without a PING
            delay(1100);
            drain();
            return "upgrade_failed_reverted";
        }
        for (const char* fast : {"espnow", "tcp", "spi"}) {
            if (strstr(p.links, fast) && try_fast_link_upgrade(p, fast)) return "upgrade_ok";
        }
        return "no_upgrade";
    }

    // HARDWARE HOOK: ESP-NOW / TCP / SPI fast links.
    // Implement when a peripheral declares one of these in manifest.bus.links:
    //   espnow: exchange MACs over UART, esp_now_add_peer(), verify PING/PONG over ESP-NOW
    //   tcp:    peripheral joins the same Wi-Fi, sends {"ip":...,"port":...}, verify over a socket
    //   spi:    needs dedicated pins (SPI3) agreed in the manifest, verify with a PING frame
    // Must return true only after a verified round-trip on the new link, and set p.link.
    static bool try_fast_link_upgrade(DeviceProfile& p, const char* link) {
        (void)p;
        Serial.printf("[UART] Peripheral offers a '%s' link; fast-link hook not implemented, staying on UART.\n", link);
        return false;
    }

    // ---- Receive path ----
    void pump_rx() {
        bool binary = state_ == LINK_RAW && (strcmp(profile_.format, "binary") == 0 || strcmp(profile_.format, "modbus_rtu") == 0);
        while (port_.available()) {
            uint8_t c = port_.read();
            last_rx_ms_ = millis();
            if (binary) {
                if (raw_len_ < sizeof(raw_)) raw_[raw_len_++] = c;
                raw_last_byte_ms_ = millis();
                continue;
            }
            if (c == '\n') {
                while (line_len_ && line_[line_len_ - 1] == '\r') line_len_--;
                line_[line_len_] = '\0';
                if (line_len_) handle_line(line_);
                line_len_ = 0;
            } else if (line_len_ < sizeof(line_) - 1) {
                line_[line_len_++] = (char)c;
            }
        }
        // Binary frames end at a 20 ms gap
        if (binary && raw_len_ && millis() - raw_last_byte_ms_ > 20) {
            char hex[2 * 64 + 1];
            size_t hn = min(raw_len_, (size_t)64);
            for (size_t i = 0; i < hn; i++) snprintf(&hex[i * 2], 3, "%02x", raw_[i]);
            hex[hn * 2] = '\0';
            observe(hex);
            raw_len_ = 0;
        }
    }

    void handle_line(const char* line) {
        if (state_ == LINK_RAW) {
            observe(line);
            return;
        }
        JsonDocument d;
        bool is_json = deserializeJson(d, line) == DeserializationError::Ok && d.is<JsonObject>();
        if (is_json) {
            if (!d["pong"].isNull()) {
                if ((d["pong"] | 0u) == ping_nonce_) {
                    ping_deadline_ms_ = 0;
                    missed_pings_ = 0;
                }
                return;
            }
            if (d["ack"].is<uint32_t>()) {
                resolve_ack(d["ack"].as<uint32_t>(), d["ok"] | true, d["err"] | "");
                return;
            }
            if (!d["fb"].isNull()) {
                JsonDocument f;
                f["kind"] = "feedback";
                f["dev"] = profile_.key;
                f["ref_act_id"] = d["id"] | 0u;
                f["value"] = d["fb"];
                log_doc(REC_FEEDBACK, f);
                return;
            }
            if (!d["id"].isNull() && !d["accepts"].isNull()) {   // the device re-introduced itself
                DeviceProfile& p = scratch_;
                if (profile_from_manifest(line, profile_.baud, p)) {
                    strlcpy(p.link, profile_.link, sizeof(p.link));
                    xSemaphoreTake(mutex_, portMAX_DELAY);
                    profile_ = p;
                    xSemaphoreGive(mutex_);
                    log_device(p, nullptr, 0);
                }
                return;
            }
        }
        submit_event(line);
    }

    void observe(const char* data) {
        if (millis() - last_observation_ms_ < EDGE_RAW_OBSERVE_INTERVAL_MS) return;
        last_observation_ms_ = millis();
        JsonDocument d;
        d["kind"] = "observation";
        d["dev"] = profile_.key;
        d["format"] = profile_.format;
        d["baud"] = profile_.baud;
        d["data"] = String(data).substring(0, 256);
        log_doc(REC_OBSERVATION, d);
    }

    void submit_event(const char* line) {
        stats_.events++;
        InFlight* slot = nullptr;
        for (auto& f : inflight_) {
            if (!f.used) {
                slot = &f;
                break;
            }
        }
        InferRequest q;
        q.act_id = next_act_id();
        q.queued_ms = millis();
        edge_build_prompt(profile_, line, q.prompt, sizeof(q.prompt));
        if (!slot || xQueueSend(infer_q_, &q, 0) != pdTRUE) {
            stats_.events_dropped++;   // inference busy: drop rather than block the UART
            return;
        }
        *slot = InFlight();
        slot->used = true;
        slot->act_id = q.act_id;
        strlcpy(slot->input, line, sizeof(slot->input));
        strlcpy(slot->prompt, q.prompt, sizeof(slot->prompt));
    }

    // ---- Results -> actions ----
    void pump_results() {
        InferResult r;
        while (xQueueReceive(result_q_, &r, 0) == pdTRUE) {
            InFlight* f = find(r.act_id);
            if (!f) continue;
            f->latency_ms = r.latency_ms;
            f->model_version = r.model_version;
            strlcpy(f->output, r.output, sizeof(f->output));
            if (!r.model_ready) {
                finish(*f, "error", "no model loaded");
                continue;
            }
            JsonDocument action;
            char why[64] = "";
            switch (validate_action(profile_, r.output, action, why, sizeof(why))) {
                case ACTION_NOP:
                    stats_.nops++;
                    finish(*f, "nop", "");
                    break;
                case ACTION_INVALID:
                    stats_.invalid++;
                    finish(*f, "invalid_output", why);
                    break;
                case ACTION_OK: {
                    serializeJson(action, f->action, sizeof(f->action));   // logged without the id
                    action["id"] = f->act_id;
                    char wire[200];
                    serializeJson(action, wire, sizeof(wire));
                    send_line(wire);
                    stats_.actions_sent++;
                    if (profile_.acks_actions) {
                        f->awaiting_ack = true;
                        f->deadline_ms = millis() + EDGE_ACTION_ACK_TIMEOUT_MS;
                    } else {
                        finish(*f, "ok", "");
                    }
                    break;
                }
            }
        }
    }

    void resolve_ack(uint32_t act_id, bool ok, const char* err) {
        InFlight* f = find(act_id);
        if (!f || !f->awaiting_ack) return;
        if (ok) stats_.acks++;
        else stats_.rejects++;
        finish(*f, ok ? "ok" : "rejected", err);
    }

    void expire_pending() {
        for (auto& f : inflight_) {
            if (f.used && f.awaiting_ack && (int32_t)(millis() - f.deadline_ms) > 0) {
                stats_.timeouts++;
                finish(f, "timeout", "no ack from peripheral");
            }
        }
    }

    void keepalive() {
        uint32_t idle = millis() - last_rx_ms_;
        if (state_ == LINK_RAW) {
            if (idle > EDGE_REDISCOVER_IDLE_MS) rediscover_ = true;
            return;
        }
        if (ping_deadline_ms_ && (int32_t)(millis() - ping_deadline_ms_) > 0) {
            ping_deadline_ms_ = 0;
            if (++missed_pings_ >= 2) {
                Serial.printf("[UART%u] '%s' stopped answering; rediscovering.\n", port_no_, profile_.key);
                rediscover_ = true;
            }
        } else if (!ping_deadline_ms_ && idle > EDGE_REDISCOVER_IDLE_MS / 3) {
            char msg[48];
            ping_nonce_ = esp_random() & 0xFFFF;
            snprintf(msg, sizeof(msg), "{\"cmd\":\"PING\",\"n\":%u}", (unsigned)ping_nonce_);
            send_line(msg);
            ping_deadline_ms_ = millis() + 1000;
            last_rx_ms_ = millis();   // next ping only after another idle period
        }
    }

    // ---- Logging ----
    InFlight* find(uint32_t act_id) {
        for (auto& f : inflight_) if (f.used && f.act_id == act_id) return &f;
        return nullptr;
    }

    void abort_inflight(const char* why) {
        for (auto& f : inflight_) if (f.used) finish(f, "error", why);
        InferResult r;
        while (result_q_ && xQueueReceive(result_q_, &r, 0) == pdTRUE) {}
    }

    void finish(InFlight& f, const char* status, const char* err) {
        JsonDocument d;
        d["kind"] = "interaction";
        d["dev"] = profile_.key;
        d["act_id"] = f.act_id;
        d["input"] = f.input;
        d["prompt"] = f.prompt;
        d["output"] = f.output;
        if (f.action[0]) d["action"] = serialized(String(f.action));
        d["status"] = status;
        if (err && *err) d["err"] = err;
        d["lat"] = f.latency_ms;
        d["mv"] = f.model_version;
        if (measureJson(d) > RING_MAX_PAYLOAD) d.remove("prompt");   // the server can rebuild it
        log_doc(REC_INTERACTION, d);
        f.used = false;
    }

    void log_device(const DeviceProfile& p, const uint8_t* sample, size_t n) {
        JsonDocument d;
        d["kind"] = "device";
        d["dev"] = p.key;
        d["port"] = port_no_;
        d["identified"] = p.identified;
        d["baud"] = p.baud;
        d["link"] = p.link;
        d["format"] = p.format;
        d["type"] = p.type;
        if (p.identified) {
            if (p.manifest_truncated) d["manifest_text"] = p.manifest;
            else d["manifest"] = serialized(String(p.manifest));
        }
        if (sample && n) {
            char hex[2 * 96 + 1];
            size_t hn = min(n, (size_t)96);
            for (size_t i = 0; i < hn; i++) snprintf(&hex[i * 2], 3, "%02x", sample[i]);
            hex[hn * 2] = '\0';
            d["sample_hex"] = hex;
        }
        log_doc(REC_DEVICE, d);
    }

    void log_doc(RingRecType type, JsonDocument& d) {
        static char buf[RING_MAX_PAYLOAD + 1];
        size_t n = serializeJson(d, buf, sizeof(buf));
        if (n == 0 || n >= sizeof(buf)) return;
        ring_->post(type, buf);
    }
};
