#pragma once
#include <Arduino.h>
#include "driver/spi_common.h"
#include "esp_flash.h"
#include "esp_flash_spi_init.h"
#include "gemma4_w25q128_config.h"

// ==============================================================================
// External W25Q128 on SPI2: detection, reads, and serial upload
// ==============================================================================
// esp_partition_mmap() only reaches the flash the ESP32-S3 boots from, so this
// chip is driven with esp_flash_* calls on SPI2 and is never memory-mapped. The
// firmware copies the model from here into PSRAM at boot.
// ==============================================================================

static esp_flash_t* g_w25 = nullptr;
static uint32_t g_w25_id = 0;     // JEDEC ID, manufacturer in the top byte (W25Q128JV = 0xEF4018)
static uint32_t g_w25_size = 0;   // bytes

// zlib-compatible CRC-32: crc32_update(0, data, len) == Python's zlib.crc32(data)
static uint32_t crc32_update(uint32_t crc, const uint8_t* data, size_t len) {
    static uint32_t table[256];
    static bool table_ready = false;
    if (!table_ready) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        table_ready = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < len; i++) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

static void w25_print_wiring() {
    Serial.printf("[W25Q128] Expected wiring (SPI2): CS=GPIO%d CLK=GPIO%d DI=GPIO%d DO=GPIO%d",
                  W25_PIN_CS, W25_PIN_CLK, W25_PIN_MOSI, W25_PIN_MISO);
    if (W25_PIN_WP >= 0 || W25_PIN_HD >= 0) Serial.printf(" WP=GPIO%d HOLD=GPIO%d", W25_PIN_WP, W25_PIN_HD);
    Serial.println(", VCC=3.3V, GND.");
    if (W25_PIN_WP < 0 || W25_PIN_HD < 0) Serial.println("[W25Q128] WP# and HOLD# must be tied to 3.3V.");
}

// Brings up SPI2 and the chip. Safe to call again after a failure (e.g. from /reload).
static bool w25_init() {
    static bool bus_ready = false;
    if (g_w25) return true;

    Serial.println("[W25Q128] Initializing external flash on SPI2...");
    if (!bus_ready) {
        spi_bus_config_t bus = {};
        bus.mosi_io_num = W25_PIN_MOSI;
        bus.miso_io_num = W25_PIN_MISO;
        bus.sclk_io_num = W25_PIN_CLK;
        bus.quadwp_io_num = W25_PIN_WP;
        bus.quadhd_io_num = W25_PIN_HD;
        bus.data4_io_num = -1;
        bus.data5_io_num = -1;
        bus.data6_io_num = -1;
        bus.data7_io_num = -1;
        esp_err_t err = spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO);
        if (err != ESP_OK) {
            Serial.printf("[W25Q128] spi_bus_initialize failed: %s\n", esp_err_to_name(err));
            return false;
        }
        bus_ready = true;
    }

    esp_flash_spi_device_config_t dev = {};
    dev.host_id = SPI2_HOST;
    dev.cs_io_num = W25_PIN_CS;
    dev.io_mode = W25_IO_MODE;
    dev.speed = W25_SPEED;
    dev.input_delay_ns = 0;
    dev.cs_id = 0;

    esp_flash_t* chip = nullptr;
    esp_err_t err = spi_bus_add_flash_device(&chip, &dev);
    if (err != ESP_OK) {
        Serial.printf("[W25Q128] spi_bus_add_flash_device failed: %s\n", esp_err_to_name(err));
        return false;
    }

    err = esp_flash_init(chip);
    if (err != ESP_OK) {
        Serial.printf("[W25Q128] No response from the chip (%s).\n", esp_err_to_name(err));
        w25_print_wiring();
        spi_bus_remove_flash_device(chip);
        return false;
    }

    esp_flash_read_id(chip, &g_w25_id);
    esp_flash_get_size(chip, &g_w25_size);
    g_w25 = chip;

    const uint8_t mfg = (g_w25_id >> 16) & 0xFF;
    Serial.printf("[W25Q128] JEDEC ID 0x%06X, %u bytes.\n", (unsigned)g_w25_id, (unsigned)g_w25_size);
    if (mfg != 0xEF) {
        Serial.println("[W25Q128] Note: manufacturer is not Winbond (0xEF); continuing as a generic SPI NOR chip.");
    }
    if (g_w25_size < W25_MODEL_HEADER_BYTES + GEMMA_MODEL_PAYLOAD_BYTES) {
        Serial.printf("[W25Q128] Warning: chip is smaller than the model image (%u bytes).\n",
                      (unsigned)(W25_MODEL_HEADER_BYTES + GEMMA_MODEL_PAYLOAD_BYTES));
    }
    return true;
}

static bool w25_read(uint32_t address, void* dst, uint32_t len) {
    return g_w25 && esp_flash_read(g_w25, dst, address, len) == ESP_OK;
}

// ------------------------------------------------------------------------------
// Serial upload, driven by gemma4/w25q128_upload.py. Protocol lines start "@W25 ".
//   host: /w25write <bytes> <crc32 hex>
//   dev : @W25 ERASE <n>/<total> ...   then   @W25 READY <chunk bytes>
//   host: <chunk>                      dev : @W25 ACK <bytes written so far>   (repeat)
//   dev : @W25 DONE <crc32>            (after reading the whole image back)
//   dev : @W25 ERR <reason>            (at any point; the transfer is over)
// ------------------------------------------------------------------------------
#define W25_UPLOAD_CHUNK 4096

// Receives an image into [base_addr, base_addr + max_len). max_len 0 = to the end of the chip.
static bool w25_receive_image(uint32_t total, uint32_t expected_crc, uint32_t base_addr = 0, uint32_t max_len = 0) {
    if (!g_w25) {
        Serial.println("@W25 ERR no W25Q128 detected");
        return false;
    }
    if (max_len == 0) max_len = g_w25_size - base_addr;
    if (total < W25_MODEL_HEADER_BYTES || total > max_len || base_addr + total > g_w25_size || (base_addr & 0xFFF)) {
        Serial.printf("@W25 ERR image size %u out of range (region holds %u bytes)\n", (unsigned)total, (unsigned)max_len);
        return false;
    }

    esp_err_t err = esp_flash_set_chip_write_protect(g_w25, false);
    if (err != ESP_OK) {
        Serial.printf("@W25 ERR could not clear write protection: %s\n", esp_err_to_name(err));
        return false;
    }

    // Erase whole 4 KB sectors covering the image, 64 KB at a time so progress is visible.
    const uint32_t BLOCK = 64 * 1024;
    const uint32_t erase_len = (total + 4095u) & ~4095u;
    const uint32_t blocks = (erase_len + BLOCK - 1) / BLOCK;
    for (uint32_t b = 0; b < blocks; b++) {
        uint32_t addr = b * BLOCK;
        uint32_t len = min(BLOCK, erase_len - addr);
        err = esp_flash_erase_region(g_w25, base_addr + addr, len);
        if (err != ESP_OK) {
            Serial.printf("@W25 ERR erase failed at 0x%06X: %s\n", (unsigned)addr, esp_err_to_name(err));
            return false;
        }
        Serial.printf("@W25 ERASE %u/%u\n", (unsigned)(b + 1), (unsigned)blocks);
    }

    static uint8_t buf[W25_UPLOAD_CHUNK];
    while (Serial.available()) Serial.read();   // drop stray bytes sent before READY
    Serial.setTimeout(5000);
    Serial.printf("@W25 READY %u\n", (unsigned)W25_UPLOAD_CHUNK);

    bool ok = true;
    uint32_t done = 0, crc = 0;
    while (done < total) {
        uint32_t n = min((uint32_t)W25_UPLOAD_CHUNK, total - done);
        size_t got = Serial.readBytes(buf, n);
        if (got != n) {
            Serial.printf("@W25 ERR timed out after %u bytes\n", (unsigned)(done + got));
            ok = false;
            break;
        }
        crc = crc32_update(crc, buf, n);
        err = esp_flash_write(g_w25, buf, base_addr + done, n);
        if (err != ESP_OK) {
            Serial.printf("@W25 ERR write failed at 0x%06X: %s\n", (unsigned)done, esp_err_to_name(err));
            ok = false;
            break;
        }
        done += n;
        Serial.printf("@W25 ACK %u\n", (unsigned)done);
    }
    Serial.setTimeout(1000);
    if (!ok) return false;

    if (crc != expected_crc) {
        Serial.printf("@W25 ERR data corrupted in transfer (CRC %08X, expected %08X)\n", (unsigned)crc, (unsigned)expected_crc);
        return false;
    }

    // Read everything back so a bad wire or a failed program shows up now, not at boot.
    uint32_t readback = 0;
    for (uint32_t addr = 0; addr < total; addr += W25_UPLOAD_CHUNK) {
        uint32_t n = min((uint32_t)W25_UPLOAD_CHUNK, total - addr);
        if (!w25_read(base_addr + addr, buf, n)) {
            Serial.printf("@W25 ERR read-back failed at 0x%06X\n", (unsigned)addr);
            return false;
        }
        readback = crc32_update(readback, buf, n);
    }
    if (readback != expected_crc) {
        Serial.printf("@W25 ERR read-back CRC %08X, expected %08X (check wiring or lower W25_SPEED)\n",
                      (unsigned)readback, (unsigned)expected_crc);
        return false;
    }

    Serial.printf("@W25 DONE %08X\n", (unsigned)readback);
    return true;
}
