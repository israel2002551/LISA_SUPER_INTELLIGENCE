#pragma once
// ==============================================================================
// Self-evolving edge node: configuration
// ==============================================================================

#include <stdint.h>

#if __has_include("secrets.h")
#include "secrets.h"
#else
#include "secrets.example.h"
#define EDGE_SECRETS_MISSING 1
#endif

#define EDGE_AI_NAME    "LISA"          // announced in the banner and the IDENTIFY handshake
#define EDGE_FW_VERSION "lisa-edge-0.1.0"

// ---- Task layout ---------------------------------------------------------------
// Core 1: inference only. Core 0: UART, flash log writer, Wi-Fi/HTTPS.
// The Core 0 GEMM helper runs at the lowest priority so I/O always preempts it:
// a busy UART or Wi-Fi slows inference down, never the other way round.
#define GEMM_WORKER_PRIORITY          1
#define EDGE_INFER_TASK_PRIORITY      5     // Core 1
#define EDGE_UART_TASK_PRIORITY       10    // Core 0
#define EDGE_RING_TASK_PRIORITY       4     // Core 0
#define EDGE_NET_TASK_PRIORITY        3     // Core 0

// ---- Peripheral UART (UART1) ---------------------------------------------------
#ifndef EDGE_UART_RX_PIN
#define EDGE_UART_RX_PIN              18
#endif
#ifndef EDGE_UART_TX_PIN
#define EDGE_UART_TX_PIN              17
#endif
#define EDGE_UART_RX_BUFFER           4096
#define EDGE_HANDSHAKE_TIMEOUT_MS     300   // wait for an IDENTIFY reply at each baud
#define EDGE_PASSIVE_LISTEN_MS        400   // listen-only window per baud for raw devices
#define EDGE_UPGRADE_TIMEOUT_MS       300   // UPGRADE ack / PING-PONG windows
#define EDGE_ACTION_ACK_TIMEOUT_MS    500   // peripheral must ack an action within this
#define EDGE_REDISCOVER_IDLE_MS       30000 // silence before the link is re-discovered
#define EDGE_NO_DEVICE_RETRY_MS       5000
#define EDGE_RAW_OBSERVE_INTERVAL_MS  10000 // log at most one raw sample per interval
#define EDGE_UART_UPGRADE_MAX_BAUD    2000000
static const uint32_t kEdgeBaudCandidates[] = {9600, 115200, 921600, 2000000};

// ---- Prompt contract -----------------------------------------------------------
// MUST match server/app/evolution/prompt_format.py, or the server trains the
// model on prompts the device never sends.
//   prompt = "<d>" + device_key + ":" + device_type + "<e>" + event[:120] + "<a>"
//   target = "NOP" | compact JSON {"cmd":...,"args":{...}}
//   With the 4,096-token vocabulary, JSON is mostly one token per character:
//   {"cmd":"set_fan","args":{"level":2}} is ~35 tokens, so actions get 48.
//   Prompt + action must fit GEMMA_MAX_CONTEXT_LENGTH (128); checked in inference_task.h.
#define EDGE_EVENT_MAX_CHARS          120
#define EDGE_CHAT_MAX_CHARS           150   // bytes of a chat message kept (server: CHAT_MAX_CHARS)
#define EDGE_CHAT_HISTORY_MAX_BYTES   160   // her last reply joins the prompt only if it all fits (CHAT_HISTORY_MAX_BYTES)
#define EDGE_CHAT_MEMORY_MS        600000   // she forgets her last reply after 10 minutes (or /new)
#define EDGE_PROMPT_MAX_TOKENS        80    // including BOS
#define EDGE_ACTION_MAX_TOKENS        48

// ---- Server sync ---------------------------------------------------------------
#define EDGE_TELEMETRY_INTERVAL_MS    15000
#define EDGE_TELEMETRY_BATCH_BYTES    12000
#define EDGE_HEARTBEAT_INTERVAL_MS    60000
#define EDGE_OTA_CHECK_INTERVAL_MS    300000
#define EDGE_HTTP_TIMEOUT_MS          15000
#define EDGE_WIFI_RETRY_MS            30000   // phone hotspots can take >10 s to accept a client

// ---- Model slots ---------------------------------------------------------------
#define EDGE_MAX_TRIAL_BOOTS          3      // reboots allowed before a new model is rolled back
#define EDGE_TRIAL_COMMIT_MS          60000  // healthy uptime before a new model is committed

// ---- USB serial (provisioning / diagnostics) ----------------------------------
#define EDGE_CONSOLE_BAUD             921600
