#pragma once
#include "esp_task_wdt.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Model updates busy-wait on the W25Q (64 KiB erases, page programs) for seconds at a
// time from the net task (Core 0, priority 3). The 1-tick yields between blocks are not
// enough to keep IDLE0 fed: during inference the GEMM worker (Core 0, priority 1) is
// also waiting to run and takes every gap, so IDLE0 can starve past the 5 s task
// watchdog and the board resets mid-update. Seen on hardware during an OTA.
//
// While an update runs, the watchdog therefore watches the task doing the work instead
// of IDLE0: a hang (no progress for 5 s) still resets the board, slow progress does not.
// Loops call wdt_feed() once per unit of progress.
class OtaWatchdogScope {
public:
    OtaWatchdogScope() {
        idle0_ = xTaskGetIdleTaskHandleForCPU(0);
        self_added_ = esp_task_wdt_add(nullptr) == ESP_OK;
        idle_removed_ = self_added_ && idle0_ && esp_task_wdt_delete(idle0_) == ESP_OK;
    }
    ~OtaWatchdogScope() {
        if (idle_removed_) esp_task_wdt_add(idle0_);
        if (self_added_) esp_task_wdt_delete(nullptr);
    }
    OtaWatchdogScope(const OtaWatchdogScope&) = delete;
    OtaWatchdogScope& operator=(const OtaWatchdogScope&) = delete;

private:
    TaskHandle_t idle0_ = nullptr;
    bool self_added_ = false;
    bool idle_removed_ = false;
};

// Inside an OtaWatchdogScope, hands the watchdog back to IDLE0 for a blocking network call
// (a TLS handshake can take longer than 5 s, and the task can't feed while it blocks).
// Outside a scope it does nothing.
class OtaWatchdogPause {
public:
    OtaWatchdogPause() {
        idle0_ = xTaskGetIdleTaskHandleForCPU(0);
        self_removed_ = esp_task_wdt_delete(nullptr) == ESP_OK;
        idle_added_ = self_removed_ && idle0_ && esp_task_wdt_add(idle0_) == ESP_OK;
    }
    ~OtaWatchdogPause() {
        if (idle_added_) esp_task_wdt_delete(idle0_);
        if (self_removed_) esp_task_wdt_add(nullptr);
    }
    OtaWatchdogPause(const OtaWatchdogPause&) = delete;
    OtaWatchdogPause& operator=(const OtaWatchdogPause&) = delete;

private:
    TaskHandle_t idle0_ = nullptr;
    bool self_removed_ = false;
    bool idle_added_ = false;
};

// Feeds the watchdog for the calling task if it is subscribed (inside an OtaWatchdogScope);
// otherwise a harmless no-op (returns ESP_ERR_NOT_FOUND).
static inline void wdt_feed() { esp_task_wdt_reset(); }
