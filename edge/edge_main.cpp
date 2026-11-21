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
#include "lisa_tokenizer_data.h"
#include "net_sync.h"

static Preferences g_prefs;
static SlotManager g_slots;
static RingLog g_ring;
static UartLink g_link(Serial1, 1, EDGE_UART_RX_PIN, EDGE_UART_TX_PIN);
static InferenceWorker g_infer;
static NetSync g_net;
static QueueHandle_t g_infer_q = nullptr;
static QueueHandle_t g_result_q = nullptr;
static QueueHandle_t g_console_q = nullptr;   // replies to /ask and /sim
static uint32_t g_boot_count = 0;
static int g_loaded_slot = -1;
static bool g_storage_ok = false;
static TaskHandle_t g_task_infer = nullptr, g_task_uart = nullptr, g_task_ring = nullptr, g_task_net = nullptr;

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
            xQueueSend(q.reply_q ? q.reply_q : g_result_q, &r, portMAX_DELAY);
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
    JsonObject st = d["stack_free_bytes"].to<JsonObject>();   // lowest free stack seen per task
    const struct { const char* name; TaskHandle_t h; } tasks[] = {
        {"infer", g_task_infer}, {"uart", g_task_uart}, {"ring", g_task_ring}, {"net", g_task_net}};
    for (const auto& t : tasks) {
        if (t.h) st[t.name] = uxTaskGetStackHighWaterMark(t.h);
    }
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

// ==============================================================================
// Talking to LISA from the USB console: /ask and /sim
// ==============================================================================
// Both go through the inference task's queue (the engine is single-task) with
// their own reply queue, so they never collide with UART traffic.

static bool console_infer(const char* prompt, uint32_t act_id, InferResult& r) {
    if (!g_model.loaded) {
        Serial.println("LISA has no model loaded (see /status).");
        return false;
    }
    InferRequest q{};
    q.act_id = act_id;
    q.queued_ms = millis();
    q.reply_q = g_console_q;
    strlcpy(q.prompt, prompt, sizeof(q.prompt));
    if (xQueueSend(g_infer_q, &q, pdMS_TO_TICKS(3000)) != pdTRUE) {
        Serial.println("LISA is busy with device events; try again.");
        return false;
    }
    Serial.print("LISA is thinking");
    for (int waited = 0; xQueueReceive(g_console_q, &r, pdMS_TO_TICKS(1000)) != pdTRUE; waited++) {
        Serial.print(".");
        if (waited > 120) {
            Serial.println(" no answer after 2 minutes.");
            return false;
        }
    }
    Serial.println();
    return true;
}

// Chat: anything typed that isn't a /command (or /ask <text>).
// She sees her previous reply (memory), and every turn is logged: the server learns
// from it, judged by your /good and /bad, her corpus, or the teacher model.
static char g_chat_prev[sizeof(InferResult::output)] = "";
static uint32_t g_chat_prev_ms = 0;
static uint32_t g_chat_count = 0;
static uint32_t g_last_chat_act = 0;   // the reply /good and /bad rate

static void ask_lisa(const char* text) {
    bool remember = g_chat_prev[0] && millis() - g_chat_prev_ms < EDGE_CHAT_MEMORY_MS;
    char prompt[sizeof(InferRequest::prompt)];
    edge_build_chat_prompt(text, remember ? g_chat_prev : nullptr, prompt, sizeof(prompt));
    uint32_t act_id = 0x40000000u | ((g_boot_count & 0x3FF) << 20) | (++g_chat_count & 0xFFFFF);
    InferResult r;
    if (!console_infer(prompt, act_id, r)) return;
    Serial.printf("LISA: %s\n      [%u ms, model v%u]\n", r.output[0] ? r.output : "(no output)", (unsigned)r.latency_ms,
                  (unsigned)r.model_version);
    if (!r.model_ready || !r.output[0]) return;
    strlcpy(g_chat_prev, r.output, sizeof(g_chat_prev));
    g_chat_prev_ms = millis();
    g_last_chat_act = act_id;

    JsonDocument d;   // no "dev": not a peripheral. status nop: no hardware was touched, no reward yet
    d["kind"] = "interaction";
    d["task"] = "chat";
    d["act_id"] = act_id;
    d["input"] = text;
    d["prompt"] = prompt;
    d["output"] = r.output;
    d["status"] = "nop";
    d["lat"] = r.latency_ms;
    d["mv"] = r.model_version;
    char buf[RING_MAX_PAYLOAD + 1];
    if (measureJson(d) > RING_MAX_PAYLOAD) d.remove("input");
    if (serializeJson(d, buf, sizeof(buf)) < sizeof(buf)) g_ring.post(REC_INTERACTION, buf);
}

// /good, /bad: your verdict on her last reply. It overrides every other judge in training.
static void rate_last_reply(bool good) {
    if (!g_last_chat_act) {
        Serial.println("Chat with LISA first, then rate her reply with /good or /bad.");
        return;
    }
    char buf[96];
    snprintf(buf, sizeof(buf), "{\"kind\":\"feedback\",\"ref_act_id\":%u,\"value\":\"%s\"}",
             (unsigned)g_last_chat_act, good ? "good" : "bad");
    g_ring.post(REC_FEEDBACK, buf);
    g_last_chat_act = 0;   // one rating per reply
    Serial.println(good ? "Thanks! She'll keep answering like that."
                        : "Noted. At her next training run she learns a better answer, if her lessons or teacher know one.");
}

// A simulated smart thermostat (same manifest and policy as tools/fake_peripheral.py).
// Its manifest carries the rules the server's teacher uses, so /sim events
// become training data and LISA learns this device's behavior.
static const char* kSimManifest =
    "{\"id\":\"sim-thermo\",\"type\":\"thermostat\",\"fw\":\"sim\",\"emits\":[\"temp\",\"hum\"],"
    "\"accepts\":[{\"cmd\":\"set_fan\",\"args\":{\"level\":\"int:0..3\"}},{\"cmd\":\"beep\",\"args\":{}}],"
    "\"needs\":[\"alerts\"],\"acks\":true,"
    "\"rules\":[{\"when\":{\"ev\":\"temp\",\"gt\":30},\"do\":{\"cmd\":\"set_fan\",\"args\":{\"level\":2}}},"
    "{\"when\":{\"ev\":\"temp\",\"lt\":15},\"do\":{\"cmd\":\"set_fan\",\"args\":{\"level\":0}}}]}";
static DeviceProfile g_sim;
static bool g_sim_ready = false;
static uint32_t g_sim_count = 0;

static void sim_setup() {
    if (g_sim_ready) return;
    profile_from_manifest(kSimManifest, 115200, g_sim);
    JsonDocument d;   // register the simulated device with the server (once per boot)
    d["kind"] = "device";
    d["dev"] = g_sim.key;
    d["port"] = 0;
    d["identified"] = true;
    d["baud"] = 115200;
    d["link"] = "uart";
    d["format"] = "json_lines";
    d["type"] = g_sim.type;
    d["manifest"] = serialized(String(kSimManifest));
    char buf[RING_MAX_PAYLOAD + 1];
    if (serializeJson(d, buf, sizeof(buf)) < sizeof(buf)) g_ring.post(REC_DEVICE, buf);
    g_sim_ready = true;
}

// Runs one temperature event through LISA. Returns true if she did the right thing.
static bool sim_event(float temp, bool verbose) {
    sim_setup();
    char event[48], prompt[200];
    snprintf(event, sizeof(event), "{\"ev\":\"temp\",\"v\":%.1f}", temp);
    edge_build_prompt(g_sim, event, prompt, sizeof(prompt));
    uint32_t act_id = 0x80000000u | ((g_boot_count & 0x7FF) << 20) | (++g_sim_count & 0xFFFFF);
    InferResult r;
    if (!console_infer(prompt, act_id, r)) return false;

    // What the thermostat wants (its own policy, like a real smart peripheral).
    int want = temp > 30 ? 2 : (temp < 15 ? 0 : -1);   // -1 = no action
    JsonDocument action;
    char why[64] = "";
    ActionVerdict v = validate_action(g_sim, r.output, action, why, sizeof(why));
    int got = (v == ACTION_OK && strcmp(action["cmd"] | "", "set_fan") == 0) ? (int)(action["args"]["level"] | -9) : -1;
    bool correct = (v == ACTION_NOP && want == -1) || (v == ACTION_OK && got == want && want != -1);
    const char* status = v == ACTION_NOP ? "nop" : (v == ACTION_INVALID ? "invalid_output" : "ok");

    // Log it exactly like a real interaction (+ the thermostat's feedback on actions).
    JsonDocument d;
    d["kind"] = "interaction";
    d["dev"] = g_sim.key;
    d["act_id"] = act_id;
    d["input"] = event;
    d["prompt"] = prompt;
    d["output"] = r.output;
    if (v == ACTION_OK) {
        char a[160];
        serializeJson(action, a, sizeof(a));
        d["action"] = serialized(String(a));
    }
    d["status"] = status;
    if (why[0]) d["err"] = why;
    d["lat"] = r.latency_ms;
    d["mv"] = r.model_version;
    char buf[RING_MAX_PAYLOAD + 1];
    if (measureJson(d) > RING_MAX_PAYLOAD) d.remove("prompt");
    if (serializeJson(d, buf, sizeof(buf)) < sizeof(buf)) g_ring.post(REC_INTERACTION, buf);
    if (v == ACTION_OK) {
        JsonDocument f;
        f["kind"] = "feedback";
        f["dev"] = g_sim.key;
        f["ref_act_id"] = act_id;
        f["value"] = correct ? "good" : "bad";
        serializeJson(f, buf, sizeof(buf));
        g_ring.post(REC_FEEDBACK, buf);
    }

    if (verbose) {
        Serial.printf("Thermostat: %s\n", event);
        Serial.printf("LISA:       %s\n", r.output[0] ? r.output : "(no output)");
        const char* expect = want == 2 ? "set_fan level 2 (hot)" : want == 0 ? "set_fan level 0 (cold)" : "NOP (comfortable)";
        Serial.printf("Verdict:    %s - wanted %s  [%u ms, model v%u]\n",
                      correct ? "CORRECT" : (v == ACTION_INVALID ? "invalid answer" : "wrong action"), expect,
                      (unsigned)r.latency_ms, (unsigned)r.model_version);
        if (v == ACTION_INVALID) Serial.printf("            (%s; nothing was sent to hardware)\n", why);
    }
    return correct;
}

// /sim <temp> | /sim auto <n>
static void simulate(const char* args) {
    while (*args == ' ') args++;
    if (strncmp(args, "auto", 4) == 0) {
        int n = constrain(atoi(args + 4), 1, 100);
        int correct = 0;
        Serial.printf("Running %d random thermostat events through LISA...\n", n);
        for (int i = 0; i < n; i++) {
            float t = (esp_random() % 451) / 10.0f;   // 0.0 .. 45.0 C
            bool ok = sim_event(t, false);
            correct += ok;
            Serial.printf("  %2d/%d  %4.1f C -> %s\n", i + 1, n, t, ok ? "correct" : "wrong");
        }
        Serial.printf("LISA got %d/%d right. These events are logged and will upload for training.\n", correct, n);
    } else if (*args) {
        sim_event(atof(args), true);
    } else {
        Serial.println("Usage: /sim <temperature>   e.g. /sim 33     or   /sim auto 20");
    }
}

static void print_help() {
    Serial.println("\n--- LISA console ---");
    Serial.println(" <any text>          Chat with LISA (or /ask <text>). She remembers her last reply.");
    Serial.println(" /good, /bad         Rate her last reply; she learns from it at the next training run");
    Serial.println(" /new                Start a new conversation (she forgets her last reply)");
    Serial.println(" /sim <temp>         Simulate a thermostat event; LISA decides, the thermostat judges");
    Serial.println(" /sim auto <n>       n random thermostat events (training data for LISA)");
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
    else if (strncmp(cmd, "/ask ", 5) == 0) ask_lisa(cmd + 5);
    else if (strcmp(cmd, "/good") == 0) rate_last_reply(true);
    else if (strcmp(cmd, "/bad") == 0) rate_last_reply(false);
    else if (strcmp(cmd, "/new") == 0) {
        g_chat_prev[0] = '\0';
        Serial.println("New conversation: LISA forgot her last reply.");
    }
    else if (strncmp(cmd, "/sim", 4) == 0) simulate(cmd + 4);
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
    else if (cmd[0] != '/') ask_lisa(cmd);
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
    if (!gemma_init_tokenizer_from(kLisaTokenizer, kLisaTokenizerBytes)) {
        Serial.println("[Init] Warning: built-in tokenizer failed to load; models get byte tokens.");
    }
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
    g_console_q = xQueueCreate(1, sizeof(InferResult));
    g_link.begin(&g_ring, g_infer_q, g_result_q, g_boot_count);
    g_net.begin(&g_slots, &g_ring, &g_link, &g_infer, &g_prefs);

    if (g_storage_ok) {
        xTaskCreatePinnedToCore(ring_task, "ring", 4096, nullptr, EDGE_RING_TASK_PRIORITY, &g_task_ring, 0);
        log_boot();
    }
    xTaskCreatePinnedToCore(inference_task, "infer", 8192, nullptr, EDGE_INFER_TASK_PRIORITY, &g_task_infer, 1);
    xTaskCreatePinnedToCore(uart_task, "uart", 12288, nullptr, EDGE_UART_TASK_PRIORITY, &g_task_uart, 0);
    // TLS handshakes (P-384 chains) run on the net task's stack.
    if (g_storage_ok) xTaskCreatePinnedToCore(net_task, "net", 16384, nullptr, EDGE_NET_TASK_PRIORITY, &g_task_net, 0);

    Serial.printf("[Init] Boot #%u ready. PSRAM free %u KB. Type /help.\n", (unsigned)g_boot_count,
                  (unsigned)(heap_caps_get_free_size(MALLOC_CAP_SPIRAM) / 1024));
}

static char s_input[EDGE_CHAT_MAX_CHARS + 32];
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
