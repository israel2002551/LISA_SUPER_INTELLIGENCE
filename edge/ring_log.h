#pragma once
// ==============================================================================
// Partition C: power-safe circular log on the W25Q128
// ==============================================================================
// Holds interaction logs, device profiles, errors and OTA events while the
// server is unreachable. Each record carries a monotonically increasing seq;
// the server acknowledges uploads by seq (cursor kept in NVS by the net task).
//
// Layout: 4 KiB sectors, each starting with {magic, first_seq}, followed by
// records {16-byte header + JSON payload padded to 4 bytes}. Records never span
// sectors. When the ring wraps, the oldest sector is erased (oldest data lost).
//
// Writers never touch flash: post() copies the record into a queue and the ring
// writer task (Core 0) does the SPI work. So a long OTA erase holding the SPI
// bus can't stall the UART task.
// ==============================================================================

#include <Arduino.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "w25q128_flash.h"
#include "w25q_layout.h"

enum RingRecType : uint8_t {
    REC_INTERACTION = 1,   // event -> model output -> action -> peripheral result
    REC_DEVICE = 2,        // discovered / identified peripheral profile
    REC_ERROR = 3,
    REC_OTA = 4,
    REC_BOOT = 5,
    REC_FEEDBACK = 6,      // peripheral feedback on an earlier action
    REC_OBSERVATION = 7,   // raw sample from an unidentified peripheral
    REC_HANDSHAKE = 8,     // handshake / bus upgrade result
};

#define RING_SECTOR_MAGIC 0x31534C52u   // "RLS1"
#define RING_REC_MAGIC    0x5AA5u
#define RING_MAX_PAYLOAD  1024
#define RING_QUEUE_DEPTH  32

struct RingSectorHeader {
    uint32_t magic;
    uint32_t first_seq;
};

struct RingRecHeader {
    uint16_t magic;
    uint8_t type;
    uint8_t flags;
    uint16_t len;        // payload bytes (before padding)
    uint16_t reserved;
    uint32_t seq;
    uint32_t crc;        // CRC-32 of the payload
};
static_assert(sizeof(RingRecHeader) == 16, "ring record header layout");

class RingLog {
public:
    // min_next_seq: high-water mark from NVS, so seq never goes backwards even if
    // the ring was wiped. Must run before any task posts.
    bool begin(uint32_t min_next_seq) {
        mutex_ = xSemaphoreCreateMutex();
        queue_ = xQueueCreate(RING_QUEUE_DEPTH, sizeof(Item));
        first_seq_ = (uint32_t*)calloc(W25_RING_SECTORS, sizeof(uint32_t));
        if (!mutex_ || !queue_ || !first_seq_) return false;

        // 1. Sector headers -> which sectors hold data and the newest one
        int newest = -1;
        for (uint32_t s = 0; s < W25_RING_SECTORS; s++) {
            RingSectorHeader h;
            if (w25_read(sector_addr(s), &h, sizeof(h)) && h.magic == RING_SECTOR_MAGIC && h.first_seq != 0) {
                first_seq_[s] = h.first_seq;
                if (newest < 0 || h.first_seq > first_seq_[newest]) newest = (int)s;
            }
        }

        if (newest < 0) {
            cur_ = W25_RING_SECTORS - 1;   // advance_sector() moves to sector 0
            next_seq_ = max(min_next_seq, (uint32_t)1);
            if (!advance_sector()) return false;
        } else {
            // 2. Walk the newest sector to find the write position and last seq
            cur_ = (uint32_t)newest;
            next_seq_ = first_seq_[cur_];
            uint32_t off = sizeof(RingSectorHeader);
            RingRecHeader r;
            while (off + sizeof(r) <= W25_SECTOR_BYTES && w25_read(sector_addr(cur_) + off, &r, sizeof(r))) {
                if (r.magic != RING_REC_MAGIC || r.len > RING_MAX_PAYLOAD || off + rec_bytes(r.len) > W25_SECTOR_BYTES) break;
                next_seq_ = r.seq + 1;
                off += rec_bytes(r.len);
            }
            write_off_ = off;
            // A torn header leaves programmed bits; never write over them.
            force_advance_ = !region_erased(sector_addr(cur_) + off, min((uint32_t)sizeof(r), W25_SECTOR_BYTES - off));
            next_seq_ = max(next_seq_, min_next_seq);
        }
        Serial.printf("[Ring] %u sectors, oldest seq %u, next seq %u.\n", (unsigned)W25_RING_SECTORS,
                      (unsigned)oldest_seq(), (unsigned)next_seq_);
        return true;
    }

    // Queues a JSON record. Safe from any task; never blocks. false = queue full (counted).
    bool post(RingRecType type, const char* json) {
        size_t len = strlen(json);
        if (len == 0 || len > RING_MAX_PAYLOAD) {
            dropped_++;
            return false;
        }
        Item it{type, (uint16_t)len, (char*)malloc(len)};
        if (!it.data) {
            dropped_++;
            return false;
        }
        memcpy(it.data, json, len);
        if (xQueueSend(queue_, &it, 0) != pdTRUE) {
            free(it.data);
            dropped_++;
            return false;
        }
        return true;
    }

    // Body of the ring writer task.
    void writer_loop() {
        Item it;
        for (;;) {
            if (xQueueReceive(queue_, &it, portMAX_DELAY) == pdTRUE) {
                xSemaphoreTake(mutex_, portMAX_DELAY);
                bool ok = append_now(it.type, (const uint8_t*)it.data, it.len);
                xSemaphoreGive(mutex_);
                if (!ok) dropped_++;
                free(it.data);
            }
        }
    }

    // Blocks until the writer has drained the queue (used before a reboot).
    void flush(uint32_t timeout_ms) {
        uint32_t t0 = millis();
        while (uxQueueMessagesWaiting(queue_) > 0 && millis() - t0 < timeout_ms) delay(10);
        delay(20);   // last item may still be mid-write
    }

    // Calls fn(seq, type, payload, len) for each intact record with seq >= from,
    // oldest first. fn returns false to stop. Returns the number of records visited.
    template <typename F>
    uint32_t read_from(uint32_t from, F fn) {
        static uint8_t payload[RING_MAX_PAYLOAD + 1];
        uint32_t visited = 0;
        xSemaphoreTake(mutex_, portMAX_DELAY);
        for (uint32_t i = 1; i <= W25_RING_SECTORS; i++) {
            uint32_t s = (cur_ + i) % W25_RING_SECTORS;   // oldest -> newest; ends at cur_
            if (first_seq_[s] == 0) continue;
            uint32_t next_first = (s == cur_) ? next_seq_ : first_seq_[(s + 1) % W25_RING_SECTORS];
            if (next_first != 0 && next_first <= from) continue;   // every record here is older than `from`

            uint32_t end = (s == cur_) ? write_off_ : W25_SECTOR_BYTES;
            uint32_t off = sizeof(RingSectorHeader);
            RingRecHeader r;
            while (off + sizeof(r) <= end && w25_read(sector_addr(s) + off, &r, sizeof(r))) {
                if (r.magic != RING_REC_MAGIC || r.len > RING_MAX_PAYLOAD || off + rec_bytes(r.len) > W25_SECTOR_BYTES) break;
                if (r.seq >= from && w25_read(sector_addr(s) + off + sizeof(r), payload, r.len) &&
                    crc32_update(0, payload, r.len) == r.crc) {
                    payload[r.len] = '\0';
                    visited++;
                    if (!fn(r.seq, r.type, (const char*)payload, r.len)) {
                        xSemaphoreGive(mutex_);
                        return visited;
                    }
                }
                off += rec_bytes(r.len);
            }
        }
        xSemaphoreGive(mutex_);
        return visited;
    }

    uint32_t next_seq() const { return next_seq_; }
    uint32_t dropped() const { return dropped_; }
    uint32_t queued() const { return uxQueueMessagesWaiting(queue_); }

    uint32_t oldest_seq() const {
        uint32_t oldest = next_seq_;
        for (uint32_t s = 0; s < W25_RING_SECTORS; s++) {
            if (first_seq_[s] != 0 && first_seq_[s] < oldest) oldest = first_seq_[s];
        }
        return oldest;
    }

private:
    struct Item {
        uint8_t type;
        uint16_t len;
        char* data;
    };

    SemaphoreHandle_t mutex_ = nullptr;
    QueueHandle_t queue_ = nullptr;
    uint32_t* first_seq_ = nullptr;   // per sector; 0 = no data
    uint32_t cur_ = 0;
    uint32_t write_off_ = 0;
    uint32_t next_seq_ = 1;
    bool force_advance_ = false;
    volatile uint32_t dropped_ = 0;

    static uint32_t sector_addr(uint32_t s) { return W25_RING_ADDR + s * W25_SECTOR_BYTES; }
    static uint32_t rec_bytes(uint32_t len) { return sizeof(RingRecHeader) + ((len + 3) & ~3u); }

    static bool region_erased(uint32_t addr, uint32_t len) {
        uint8_t buf[16];
        if (len == 0) return true;
        if (!w25_read(addr, buf, len)) return false;
        for (uint32_t i = 0; i < len; i++) if (buf[i] != 0xFF) return false;
        return true;
    }

    // Erases the next sector (dropping its old records) and stamps its header.
    bool advance_sector() {
        cur_ = (cur_ + 1) % W25_RING_SECTORS;
        first_seq_[cur_] = 0;
        RingSectorHeader h{RING_SECTOR_MAGIC, next_seq_};
        if (esp_flash_erase_region(g_w25, sector_addr(cur_), W25_SECTOR_BYTES) != ESP_OK ||
            esp_flash_write(g_w25, &h, sector_addr(cur_), sizeof(h)) != ESP_OK) {
            return false;
        }
        first_seq_[cur_] = next_seq_;
        write_off_ = sizeof(RingSectorHeader);
        force_advance_ = false;
        return true;
    }

    bool append_now(uint8_t type, const uint8_t* data, uint16_t len) {
        static uint8_t buf[sizeof(RingRecHeader) + RING_MAX_PAYLOAD + 4];
        uint32_t total = rec_bytes(len);
        if (force_advance_ || write_off_ + total > W25_SECTOR_BYTES) {
            if (!advance_sector()) return false;
        }
        RingRecHeader r{(uint16_t)RING_REC_MAGIC, type, 0, len, 0, next_seq_, crc32_update(0, data, len)};
        memset(buf, 0xFF, total);
        memcpy(buf, &r, sizeof(r));
        memcpy(buf + sizeof(r), data, len);
        if (esp_flash_write(g_w25, buf, sector_addr(cur_) + write_off_, total) != ESP_OK) {
            force_advance_ = true;   // don't reuse a half-programmed area
            return false;
        }
        write_off_ += total;
        next_seq_++;
        return true;
    }
};
