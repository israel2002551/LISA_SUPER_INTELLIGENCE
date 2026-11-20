// ==============================================================================
// LISA: self-evolving edge node (ESP32-S3 + external W25Q128 + Gemma 4 INT16)
// ==============================================================================
// Core 1  inference_task   local Gemma inference (the only engine user)
// Core 0  uart_task        UART auto-baud / IDENTIFY / bus upgrade / actions
//         ring_task        writes log records to Partition C of the W25Q128
//         net_task         Wi-Fi, telemetry, OTA (A/B slots), cloud classification
//         GEMM helper      Core 0 half of the matmuls, lowest priority
// loop()  (Core 1, low priority) USB console + trial-model commit
//
// Bring-up order and per-module tests: docs/BRINGUP.md
// ==============================================================================

#include "edge_config.h"     // must come first: sets GEMM_WORKER_PRIORITY for the engine
#include "gemma4_engine.h"
#include <Preferences.h>
#include "w25q_layout.h"
#include "slot_manager.h"
#include "ring_log.h"
#include "action_schema.h"
#include "edge_messages.h"
#include "uart_link.h"
#include "inference_task.h"
#include "net_sync.h"

static Preferences g_prefs;
static SlotManager g_slots;
static RingLog g_ring;
static UartLink g_link(Serial1, 1, EDGE_UART_RX_PIN, EDGE_UART_TX_PIN);
static InferenceWorker g_infer;
static NetSync g_net;
static QueueHandle_t g_infer_q = nullptr;
static QueueHandle_t g_result_q = nullptr;
static uint32_t g_boot_count = 0;
static int g_loaded_slot = -1;
static bool g_storage_ok = false;

// ==============================================================================
// Tasks
// ==============================================================================
static void inference_task(void*) {
    g_infer.self_test();
    InferRequest q;
    InferResult r;
    for (;;) {
        if (xQueueReceive(g_infer_q, &q, portMAX_DELAY) == pdTRUE) {
            g_infer.run(q, r);
            xQueueSend(g_result_q, &r, portMAX_DELAY);
        }
    }
}

static void uart_task(void*) {
    for (;;) g_link.step();
}

static void ring_task(void*) { g_ring.writer_loop(); }

static void net_task(void*) {
    for (;;) g_net.step();
}

// ==============================================================================
// Boot: pick a slot, load it, verify it against the boot record
// ==============================================================================
static void boot_model() {
    int s = g_slots.select_boot_slot();
    while (s >= 0) {
        uint8_t sha[32];
        const SlotInfo& info = g_slots.info(s);
        Serial.printf("[Boot] Loading slot %c (model v%u, %s).\n", 'A' + s, (unsigned)info.version, slot_state_name(info.state));
        if (!gemma_load_model(w25_slot_addr(s), W25_SLOT_BYTES, sha)) {
            s = g_slots.report_load_failure(s, "image failed to load");
            continue;
        }
        if (memcmp(sha, info.sha256, 32) != 0) {
            gemma_unload_model();
            s = g_slots.report_load_failure(s, "SHA-256 differs from the boot record");
            continue;
        }
        g_loaded_slot = s;
        return;
    }
    Serial.println("[Boot] No usable model. Provision one over USB: python gemma4/w25q128_upload.py <port> <image>");
}

static void log_boot() {
    JsonDocument d;
    d["kind"] = "boot";
    d["boot"] = g_boot_count;
    d["reset_reason"] = (int)esp_reset_reason();
    d["fw"] = EDGE_FW_VERSION;
    d["slot"] = g_loaded_slot < 0 ? "none" : (g_loaded_slot == 0 ? "A" : "B");
    d["model_version"] = g_model.model_version;
    d["model_loaded"] = g_model.loaded;
    char buf[256];
    serializeJson(d, buf, sizeof(buf));
    g_ring.post(REC_BOOT, buf);
}

// ==============================================================================
// USB console
// ==============================================================================
static void print_status() {
    JsonDocument d;
    d["fw"] = EDGE_FW_VERSION;
    d["node"] = EDGE_NODE_ID;
    d["boot"] = g_boot_count;
    d["uptime_s"] = millis() / 1000;
    d["model_loaded"] = g_model.loaded;
    d["model_version"] = g_model.model_version;
    d["selftest_ok"] = g_infer.selftest_ok;
    d["wifi"] = g_net.configured() ? (g_net.online() ? "online" : "offline") : "not configured";
    if (g_storage_ok) g_slots.describe(d["partitions"].to<JsonObject>());
    JsonObject r = d["ring"].to<JsonObject>();
    r["next_seq"] = g_ring.next_seq();
    r["oldest_seq"] = g_ring.oldest_seq();
    r["ack_next"] = g_net.cursor();
    r["dropped"] = g_ring.dropped();
    DeviceProfile p;
    g_link.snapshot(p);
    JsonObject u = d["uart1"].to<JsonObject>();
    u["state"] = link_state_name(g_link.state());
    u["device"] = p.key;
    u["type"] = p.type;
    u["baud"] = p.baud;
    u["link"] = p.link;
    u["format"] = p.format;
    u["commands"] = p.n_accepts;
    serializeJsonPretty(d, Serial);
    Serial.println();
}

static void print_ring(uint32_t count) {
    uint32_t from = g_ring.next_seq() > count ? g_ring.next_seq() - count : 1;
    g_ring.read_from(from, [](uint32_t seq, uint8_t type, const char* payload, size_t) {
        Serial.printf("#%u [%u] %s\n", (unsigned)seq, type, payload);
        return true;
    });
}

// Writes an image received over USB into the inactive slot and boots it.
// Operator-provisioned images are trusted (GOOD) immediately.
static void provision_over_usb(const char* args) {
    unsigned long size = 0, crc = 0;
    if (sscanf(args, "%lu %lx", &size, &crc) != 2) {
        Serial.println("@W25 ERR usage: /w25write <bytes> <crc32 hex>");
        return;
    }
    if (!g_storage_ok || !g_slots.try_lock_staging(pdMS_TO_TICKS(1000))) {
        Serial.println("@W25 ERR storage busy or unavailable");
        return;
    }
    int slot = g_slots.staging();
    bool ok = g_slots.invalidate_staging() && w25_receive_image((uint32_t)size, (uint32_t)crc, w25_slot_addr(slot), W25_SLOT_BYTES);
    if (ok) {
        W25ModelHeader h;
        uint8_t sha[32];
        ok = g_slots.read_slot(slot, 0, &h, sizeof(h)) && gemma_read_image_header(w25_slot_addr(slot), h) &&
             g_slots.hash_slot(slot, (uint32_t)size, sha) && g_slots.finalize_staging(h.model_version, (uint32_t)size, sha, true);
    }
    g_slots.unlock_staging();
    if (!ok) {
        Serial.println("@W25 ERR image written but not accepted (see above)");
        return;
    }
    Serial.printf("@W25 LOADED into slot %c; rebooting\n", 'A' + slot);
    g_ring.flush(2000);
    delay(200);
    ESP.restart();
}

static void print_help() {
    Serial.println("\n--- Edge node console ---");
    Serial.println(" /status             Model, slots, ring log, UART link, Wi-Fi");
    Serial.println(" /ring [n]           Print the last n ring-log records (default 10)");
    Serial.println(" /rediscover         Re-run UART auto-baud + IDENTIFY");
    Serial.println(" /ota                Check the server for a new model now");
    Serial.println(" /commit             Accept the trial model now");
    Serial.println(" /rollback           Reject the active model and boot the other slot");
    Serial.println(" /w25write <n> <crc> Receive a model image into the inactive slot (w25q128_upload.py)\n");
}

static void handle_command(const char* cmd) {
    if (strcmp(cmd, "/help") == 0) print_help();
    else if (strcmp(cmd, "/status") == 0) print_status();
    else if (strncmp(cmd, "/ring", 5) == 0) print_ring(cmd[5] ? max(1, atoi(cmd + 6)) : 10);
    else if (strcmp(cmd, "/rediscover") == 0) g_link.request_rediscover();
    else if (strcmp(cmd, "/ota") == 0) g_net.request_ota_check();
    else if (strcmp(cmd, "/commit") == 0) Serial.println(g_slots.commit_trial() ? "Committed." : "Active slot is not on trial.");
    else if (strcmp(cmd, "/rollback") == 0) {
        if (g_slots.rollback()) {
            Serial.println("Rolled back; rebooting.");
            g_ring.flush(2000);
            ESP.restart();
        } else {
            Serial.println("No GOOD model in the other slot.");
        }
    } else if (strncmp(cmd, "/w25write ", 10) == 0) provision_over_usb(cmd + 10);
    else Serial.println("Unknown command. Type /help.");
}

// ==============================================================================
// Setup & loop
// ==============================================================================
void setup() {
    Serial.setRxBufferSize(2 * W25_UPLOAD_CHUNK);
    Serial.begin(EDGE_CONSOLE_BAUD);
    delay(1500);
    Serial.println("\n===============================================================");
    Serial.printf("   %s - self-evolving edge AI (ESP32-S3 + W25Q128 + Gemma 4)\n", EDGE_AI_NAME);
    Serial.printf("   node %s, firmware %s\n", EDGE_NODE_ID, EDGE_FW_VERSION);
    Serial.println("===============================================================");
#ifdef EDGE_SECRETS_MISSING
    Serial.println("[Init] edge/secrets.h not found: running offline (no Wi-Fi, no OTA).");
#endif

    g_prefs.begin("edge", false);
    g_boot_count = g_prefs.getUInt("boots", 0) + 1;
    g_prefs.putUInt("boots", g_boot_count);

    if (!psramFound()) {
        Serial.println("[FATAL] No PSRAM. This build needs an ESP32-S3 with PSRAM (e.g. N16R8). Halting.");
        while (1) delay(1000);
    }
    gemma_start_gemm_worker(GEMM_WORKER_PRIORITY);
    if (!gemma_init_tokenizer()) Serial.println("[Init] Warning: 'tokenizer' partition missing; using byte tokens.");
    if (!g_cache_mgr.kv.init(GEMMA_MAX_CONTEXT_LENGTH)) {
        Serial.println("[FATAL] KV cache allocation failed. Halting.");
        while (1) delay(1000);
    }

    // External flash: boot control + slots + ring log
    g_storage_ok = w25_init() && g_slots.begin() && g_ring.begin(g_prefs.getUInt("seq_hw", 1));
    if (g_storage_ok) {
        boot_model();
    } else {
        Serial.println("[Init] W25Q128 unavailable: no model, no offline log. Check wiring (see /help).");
        w25_print_wiring();
    }

    g_infer_q = xQueueCreate(4, sizeof(InferRequest));
    g_result_q = xQueueCreate(8, sizeof(InferResult));
    g_link.begin(&g_ring, g_infer_q, g_result_q, g_boot_count);
    g_net.begin(&g_slots, &g_ring, &g_link, &g_infer, &g_prefs);

    if (g_storage_ok) {
        xTaskCreatePinnedToCore(ring_task, "ring", 4096, nullptr, EDGE_RING_TASK_PRIORITY, nullptr, 0);
        log_boot();
    }
    xTaskCreatePinnedToCore(inference_task, "infer", 8192, nullptr, EDGE_INFER_TASK_PRIORITY, nullptr, 1);
    xTaskCreatePinnedToCore(uart_task, "uart", 6144, nullptr, EDGE_UART_TASK_PRIORITY, nullptr, 0);
    if (g_storage_ok) xTaskCreatePinnedToCore(net_task, "net", 12288, nullptr, EDGE_NET_TASK_PRIORITY, nullptr, 0);

    Serial.printf("[Init] Boot #%u ready. PSRAM free %u KB. Type /help.\n", (unsigned)g_boot_count,
                  (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
}

static char s_input[128];
static int s_input_len = 0;

void loop() {
    // Trial model: commit after a healthy window, roll back if the self-test failed.
    if (g_loaded_slot >= 0 && g_infer.selftest_done && g_slots.info(g_loaded_slot).state == SLOT_TRIAL) {
        if (!g_infer.selftest_ok) {
            g_slots.report_load_failure(g_loaded_slot, "self-test failed");
            g_ring.flush(2000);
            ESP.restart();
        } else if (millis() > EDGE_TRIAL_COMMIT_MS) {
            g_slots.commit_trial();
            g_ring.post(REC_OTA, "{\"kind\":\"ota\",\"stage\":\"committed\"}");
        }
    }

    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\r' || c == '\n') {
            if (s_input_len) {
                s_input[s_input_len] = '\0';
                Serial.println();
                handle_command(s_input);
                s_input_len = 0;
            }
        } else if (s_input_len < (int)sizeof(s_input) - 1) {
            s_input[s_input_len++] = c;
            Serial.print(c);
        }
    }
    delay(20);
}
