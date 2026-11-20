#pragma once
// ==============================================================================
// A/B model slots on the W25Q128 with atomic switching and rollback
// ==============================================================================
// Boot control lives in two 4 KiB sectors (ping-pong). Each write goes to the
// sector NOT holding the current record, with seq+1 and a CRC-32; at boot the
// valid copy with the higher seq wins. A power cut mid-write leaves the old
// record intact, so switching the active slot is atomic.
//
// Slot lifecycle:
//   EMPTY -> (download into staging slot, SHA-256 verified) -> TRIAL (active)
//   TRIAL -> GOOD   after EDGE_TRIAL_COMMIT_MS of healthy uptime + self-test
//   TRIAL -> BAD    after EDGE_MAX_TRIAL_BOOTS boots without a commit, or a
//                   failed load/self-test; the other GOOD slot becomes active.
// ==============================================================================

#include <Arduino.h>
#include <ArduinoJson.h>
#include <stddef.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "w25q128_flash.h"
#include "sha256_util.h"
#include "w25q_layout.h"
#include "edge_config.h"

enum SlotState : uint32_t {
    SLOT_EMPTY = 0,
    SLOT_STAGED = 1,
    SLOT_TRIAL = 2,
    SLOT_GOOD = 3,
    SLOT_BAD = 4,
};

static const char* slot_state_name(uint32_t s) {
    switch (s) {
        case SLOT_EMPTY: return "empty";
        case SLOT_STAGED: return "staged";
        case SLOT_TRIAL: return "trial";
        case SLOT_GOOD: return "good";
        case SLOT_BAD: return "bad";
        default: return "?";
    }
}

struct SlotInfo {
    uint32_t state;
    uint32_t version;     // model_versions.version (0 = unversioned)
    uint32_t size;        // image bytes (header + payload)
    uint8_t sha256[32];   // SHA-256 of those bytes
};

#define BOOT_CTRL_MAGIC 0x31544342u   // "BCT1"

struct BootControl {
    uint32_t magic;
    uint32_t seq;
    uint32_t active;          // 0 = slot A, 1 = slot B
    uint32_t boot_attempts;   // boots of the active slot while it is TRIAL
    SlotInfo slot[2];
    uint32_t crc32;           // CRC-32 of every byte above
};

class SlotManager {
public:
    BootControl ctl{};

    bool begin() {
        if (!mutex_) mutex_ = xSemaphoreCreateRecursiveMutex();
        if (!staging_lock_) staging_lock_ = xSemaphoreCreateMutex();
        Lock l(mutex_);

        BootControl c0, c1;
        bool v0 = read_copy(W25_CTRL_ADDR_0, c0);
        bool v1 = read_copy(W25_CTRL_ADDR_1, c1);
        if (v0 && (!v1 || c0.seq >= c1.seq)) {
            ctl = c0;
            copy_ = 0;
        } else if (v1) {
            ctl = c1;
            copy_ = 1;
        } else {
            // Fresh chip (or first boot of this firmware): adopt valid images already in the slots.
            Serial.println("[Slots] No boot control record; scanning slots.");
            memset(&ctl, 0, sizeof(ctl));
            ctl.magic = BOOT_CTRL_MAGIC;
            copy_ = 1;   // first persist() goes to copy 0
            for (int s = 0; s < 2; s++) adopt_existing(s);
            ctl.active = (ctl.slot[0].state != SLOT_GOOD && ctl.slot[1].state == SLOT_GOOD) ? 1 : 0;
            if (!persist()) return false;
        }
        ctl.active &= 1;
        return true;
    }

    int active() const { return (int)(ctl.active & 1); }
    int staging() const { return 1 - active(); }
    const SlotInfo& info(int slot) const { return ctl.slot[slot & 1]; }

    // Picks the slot to load at boot. Counts trial boots and rolls back when a
    // trial slot keeps failing. Returns -1 if no slot holds a usable image.
    int select_boot_slot() {
        Lock l(mutex_);
        int a = active();
        if (ctl.slot[a].state == SLOT_TRIAL) {
            ctl.boot_attempts++;
            Serial.printf("[Slots] Slot %c is on trial (boot %u of %u).\n", 'A' + a, (unsigned)ctl.boot_attempts, EDGE_MAX_TRIAL_BOOTS);
            if (ctl.boot_attempts > EDGE_MAX_TRIAL_BOOTS) {
                Serial.printf("[Slots] Slot %c never got healthy; marking it bad.\n", 'A' + a);
                ctl.slot[a].state = SLOT_BAD;
                switch_to_other_good();
            }
            persist();
        }
        a = active();
        if (usable(a)) return a;
        if (switch_to_other_good()) {
            persist();
            return active();
        }
        return -1;
    }

    // The active slot failed to load or verify: mark it bad and fall back.
    // Returns the slot to try next, or -1.
    int report_load_failure(int slot, const char* why) {
        Lock l(mutex_);
        Serial.printf("[Slots] Slot %c rejected: %s\n", 'A' + slot, why);
        ctl.slot[slot & 1].state = SLOT_BAD;
        int next = -1;
        if (active() == slot && switch_to_other_good()) next = active();
        persist();
        return next;
    }

    // TRIAL -> GOOD once the new model has proven itself.
    bool commit_trial() {
        Lock l(mutex_);
        int a = active();
        if (ctl.slot[a].state != SLOT_TRIAL) return false;
        ctl.slot[a].state = SLOT_GOOD;
        ctl.boot_attempts = 0;
        Serial.printf("[Slots] Committed slot %c (model v%u).\n", 'A' + a, (unsigned)ctl.slot[a].version);
        return persist();
    }

    // Reject the active model and go back to the other slot (server- or operator-requested).
    bool rollback() {
        Lock l(mutex_);
        int a = active();
        if (ctl.slot[1 - a].state != SLOT_GOOD) return false;
        ctl.slot[a].state = SLOT_BAD;
        switch_to_other_good();
        return persist();
    }

    // ---- Staging (OTA download or serial provisioning into the inactive slot) ----
    bool try_lock_staging(TickType_t wait = 0) { return xSemaphoreTake(staging_lock_, wait) == pdTRUE; }
    void unlock_staging() { xSemaphoreGive(staging_lock_); }

    // Marks the staging slot EMPTY so a half-written slot is never trusted.
    bool invalidate_staging() {
        Lock l(mutex_);
        ctl.slot[staging()].state = SLOT_EMPTY;
        return persist();
    }

    // Marks the staging slot EMPTY, then erases it.
    bool erase_staging(uint32_t size) {
        if (size == 0 || size > W25_SLOT_BYTES) return false;
        int s = staging();
        {
            Lock l(mutex_);
            ctl.slot[s].state = SLOT_EMPTY;
            if (!persist()) return false;
        }
        const uint32_t BLOCK = 0x10000;
        uint32_t len = (size + BLOCK - 1) & ~(BLOCK - 1);
        for (uint32_t off = 0; off < len; off += BLOCK) {
            if (esp_flash_erase_region(g_w25, w25_slot_addr(s) + off, BLOCK) != ESP_OK) return false;
            vTaskDelay(1);   // let lower-priority Core 0 tasks run between 64 KiB erases
        }
        return true;
    }

    bool write_staging(uint32_t offset, const void* data, size_t len) {
        if (offset + len > W25_SLOT_BYTES) return false;
        return esp_flash_write(g_w25, data, w25_slot_addr(staging()) + offset, len) == ESP_OK;
    }

    bool read_slot(int slot, uint32_t offset, void* dst, size_t len) {
        if (offset + len > W25_SLOT_BYTES) return false;
        return w25_read(w25_slot_addr(slot) + offset, dst, len);
    }

    // SHA-256 of the first `size` bytes of a slot, read back from the chip.
    bool hash_slot(int slot, uint32_t size, uint8_t out[32]) {
        static uint8_t buf[4096];
        Sha256 sha;
        for (uint32_t off = 0; off < size; off += sizeof(buf)) {
            uint32_t n = min((uint32_t)sizeof(buf), size - off);
            if (!read_slot(slot, off, buf, n)) return false;
            sha.update(buf, n);
        }
        sha.finish(out);
        return true;
    }

    // Verifies the staged bytes against `sha256` and makes the staging slot active.
    // trusted = operator-provisioned over USB (GOOD at once); otherwise TRIAL.
    bool finalize_staging(uint32_t version, uint32_t size, const uint8_t sha256[32], bool trusted) {
        int s = staging();
        uint8_t actual[32];
        if (!hash_slot(s, size, actual) || memcmp(actual, sha256, 32) != 0) {
            Serial.println("[Slots] Staged image failed SHA-256 verification.");
            return false;
        }
        Lock l(mutex_);
        ctl.slot[s].state = trusted ? SLOT_GOOD : SLOT_TRIAL;
        ctl.slot[s].version = version;
        ctl.slot[s].size = size;
        memcpy(ctl.slot[s].sha256, sha256, 32);
        ctl.active = s;
        ctl.boot_attempts = 0;
        Serial.printf("[Slots] Slot %c now active (model v%u, %s).\n", 'A' + s, (unsigned)version, slot_state_name(ctl.slot[s].state));
        return persist();
    }

    void active_sha_hex(char out[65]) {
        Lock l(mutex_);
        sha256_to_hex(ctl.slot[active()].sha256, out);
    }

    void describe(JsonObject out) {
        Lock l(mutex_);
        out["active_slot"] = active();
        out["boot_attempts"] = ctl.boot_attempts;
        out["seq"] = ctl.seq;
        JsonArray arr = out["slots"].to<JsonArray>();
        for (int s = 0; s < 2; s++) {
            JsonObject o = arr.add<JsonObject>();
            char hex[65];
            sha256_to_hex(ctl.slot[s].sha256, hex);
            o["slot"] = s == 0 ? "A" : "B";
            o["state"] = slot_state_name(ctl.slot[s].state);
            o["version"] = ctl.slot[s].version;
            o["size"] = ctl.slot[s].size;
            // String() forces a copy: ArduinoJson keeps a bare const char* by pointer,
            // and `hex` is reused on the next loop pass.
            o["sha256"] = String(ctl.slot[s].state == SLOT_EMPTY ? "" : hex);
        }
    }

private:
    SemaphoreHandle_t mutex_ = nullptr;
    SemaphoreHandle_t staging_lock_ = nullptr;
    int copy_ = 0;   // which control sector holds `ctl`

    struct Lock {
        SemaphoreHandle_t m;
        explicit Lock(SemaphoreHandle_t mm) : m(mm) { xSemaphoreTakeRecursive(m, portMAX_DELAY); }
        ~Lock() { xSemaphoreGiveRecursive(m); }
    };

    static uint32_t record_crc(const BootControl& c) {
        return crc32_update(0, (const uint8_t*)&c, offsetof(BootControl, crc32));
    }

    static bool read_copy(uint32_t addr, BootControl& out) {
        return w25_read(addr, &out, sizeof(out)) && out.magic == BOOT_CTRL_MAGIC && out.crc32 == record_crc(out);
    }

    bool usable(int s) const { return ctl.slot[s].state == SLOT_GOOD || ctl.slot[s].state == SLOT_TRIAL; }

    bool switch_to_other_good() {
        int o = 1 - active();
        if (ctl.slot[o].state != SLOT_GOOD) return false;
        ctl.active = o;
        ctl.boot_attempts = 0;
        Serial.printf("[Slots] Rolled back to slot %c (model v%u).\n", 'A' + o, (unsigned)ctl.slot[o].version);
        return true;
    }

    // Writes ctl (seq+1) to the other control sector and reads it back.
    bool persist() {
        ctl.seq++;
        ctl.crc32 = record_crc(ctl);
        uint32_t addr = (copy_ == 0) ? W25_CTRL_ADDR_1 : W25_CTRL_ADDR_0;
        BootControl check;
        if (esp_flash_erase_region(g_w25, addr, W25_SECTOR_BYTES) != ESP_OK ||
            esp_flash_write(g_w25, &ctl, addr, sizeof(ctl)) != ESP_OK ||
            !read_copy(addr, check) || check.seq != ctl.seq) {
            Serial.println("[Slots] ERROR: could not write the boot control record.");
            return false;
        }
        copy_ = (addr == W25_CTRL_ADDR_0) ? 0 : 1;
        return true;
    }

    // Marks a slot GOOD if it already holds a valid image (e.g. provisioned before this firmware).
    void adopt_existing(int s) {
        W25ModelHeader h;
        if (!read_slot(s, 0, &h, sizeof(h)) || h.magic != W25_MODEL_MAGIC || h.header_bytes != W25_MODEL_HEADER_BYTES ||
            h.payload_bytes != GEMMA_MODEL_PAYLOAD_BYTES) {
            ctl.slot[s].state = SLOT_EMPTY;
            return;
        }
        uint32_t size = W25_MODEL_HEADER_BYTES + h.payload_bytes;
        if (!hash_slot(s, size, ctl.slot[s].sha256)) {
            ctl.slot[s].state = SLOT_EMPTY;
            return;
        }
        ctl.slot[s].state = SLOT_GOOD;
        ctl.slot[s].version = h.model_version;
        ctl.slot[s].size = size;
        Serial.printf("[Slots] Adopted existing image in slot %c (model v%u).\n", 'A' + s, (unsigned)h.model_version);
    }
};
