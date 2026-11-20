#pragma once
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

// Messages between the UART task (Core 0) / USB console and the inference task
// (Core 1). Fixed-size so they can travel through FreeRTOS queues by copy.

struct InferRequest {
    uint32_t act_id;         // correlates request, result, peripheral ack and feedback
    uint32_t queued_ms;
    QueueHandle_t reply_q;   // where the result goes; nullptr = the UART task's result queue
    char prompt[200];        // already in the prompt contract format (action_schema.h)
};

struct InferResult {
    uint32_t act_id;
    uint32_t latency_ms;     // queue wait + inference
    uint32_t model_version;
    bool model_ready;        // false: no model loaded, output is empty
    char output[160];        // decoded text, first line only
};
