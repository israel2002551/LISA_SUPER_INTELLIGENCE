#pragma once
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <Arduino.h>
#include "gemma4_w25q128_config.h"

// FreeRTOS Task Synchronization for Dual-Core Xtensa LX7
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// ==============================================================================
// Gemma 4 INT16 (Q16) Quantized Architecture Specification
// ==============================================================================
// - Precision: Signed 16-bit integers (int16_t, [-32768, +32767])
// - Scaling: Per-row Float32 scale factors (w_float = w_int16 * scale)
// - Dynamic Range: 96 dB (vs 24 dB of INT4), preserving GeGLU & attention precision
// - Accelerators: Dual-Core Xtensa LX7 16-bit MAC parallelization
// ==============================================================================

// Forward declarations
struct GemmaQ16Matrix;

// Dual-Core Parallel GEMM Work Sharing Primitives
struct ParallelQ16GemmTask {
    const int16_t* weights;
    const float* scales;
    const float* x;
    float* xout;
    uint32_t in_dim;
    uint32_t start_row;
    uint32_t end_row;
};

static volatile ParallelQ16GemmTask g_core0_q16_task;
static TaskHandle_t g_core0_q16_handle = NULL;
static TaskHandle_t g_core1_q16_handle = NULL;
static volatile bool g_core0_q16_ready = false;

// Core 0 Worker Task: Computes lower half of INT16 matrix rows in parallel
static void core0_q16_gemm_worker(void* pvParameters) {
    g_core1_q16_handle = (TaskHandle_t)pvParameters;
    g_core0_q16_ready = true;

    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        const int16_t* weights = g_core0_q16_task.weights;
        const float* scales = g_core0_q16_task.scales;
        const float* x = g_core0_q16_task.x;
        float* xout = g_core0_q16_task.xout;
        uint32_t in_dim = g_core0_q16_task.in_dim;

        for (uint32_t i = g_core0_q16_task.start_row; i < g_core0_q16_task.end_row; i++) {
            const int16_t* row = &weights[i * in_dim];
            float sum = 0.0f;
            uint32_t j = 0;
            // 4x unrolled inner loop for dual-issue Xtensa pipeline
            for (; j + 3 < in_dim; j += 4) {
                sum += (float)row[j] * x[j]
                     + (float)row[j + 1] * x[j + 1]
                     + (float)row[j + 2] * x[j + 2]
                     + (float)row[j + 3] * x[j + 3];
            }
            for (; j < in_dim; j++) {
                sum += (float)row[j] * x[j];
            }
            xout[i] = sum * scales[i];
        }

        xTaskNotifyGive(g_core1_q16_handle);
    }
}

// ==============================================================================
// GemmaQ16Matrix: 2D Matrix with INT16 Weights and FP32 Row Scales
// ==============================================================================
struct GemmaQ16Matrix {
    uint32_t out_dim;
    uint32_t in_dim;
    const float* scales;      // [out_dim] float32 scales
    const int16_t* weights;   // [out_dim * in_dim] signed 16-bit integers

    static GemmaQ16Matrix from_buffer(const uint8_t* base, uint32_t out_d, uint32_t in_d, uint32_t& offset) {
        GemmaQ16Matrix m;
        m.out_dim = out_d;
        m.in_dim = in_d;
        m.scales = (const float*)(base + offset);
        offset += out_d * sizeof(float);
        m.weights = (const int16_t*)(base + offset);
        offset += out_d * in_d * sizeof(int16_t);
        return m;
    }

    // Single-core fallback / general matmul
    inline void matmul(float* xout, const float* x) const {
        for (uint32_t i = 0; i < out_dim; i++) {
            const int16_t* row = &weights[i * in_dim];
            float sum = 0.0f;
            uint32_t j = 0;
            for (; j + 3 < in_dim; j += 4) {
                sum += (float)row[j] * x[j]
                     + (float)row[j + 1] * x[j + 1]
                     + (float)row[j + 2] * x[j + 2]
                     + (float)row[j + 3] * x[j + 3];
            }
            for (; j < in_dim; j++) {
                sum += (float)row[j] * x[j];
            }
            xout[i] = sum * scales[i];
        }
    }

    // Dual-core accelerated parallel matmul across Core 0 and Core 1
    inline void matmul_parallel(float* xout, const float* x) const {
        if (!g_core0_q16_ready || out_dim < 32) {
            matmul(xout, x);
            return;
        }

        uint32_t half_out = out_dim / 2;

        // Dispatch lower half (0 .. half_out - 1) to Core 0
        g_core0_q16_task.weights = weights;
        g_core0_q16_task.scales = scales;
        g_core0_q16_task.x = x;
        g_core0_q16_task.xout = xout;
        g_core0_q16_task.in_dim = in_dim;
        g_core0_q16_task.start_row = 0;
        g_core0_q16_task.end_row = half_out;

        // The worker notifies whichever task is waiting here (setup's task in the
        // chat build, the inference task in the edge build).
        g_core1_q16_handle = xTaskGetCurrentTaskHandle();
        xTaskNotifyGive(g_core0_q16_handle);

        // Core 1 immediately computes upper half (half_out .. out_dim - 1)
        for (uint32_t i = half_out; i < out_dim; i++) {
            const int16_t* row = &weights[i * in_dim];
            float sum = 0.0f;
            uint32_t j = 0;
            for (; j + 3 < in_dim; j += 4) {
                sum += (float)row[j] * x[j]
                     + (float)row[j + 1] * x[j + 1]
                     + (float)row[j + 2] * x[j + 2]
                     + (float)row[j + 3] * x[j + 3];
            }
            for (; j < in_dim; j++) {
                sum += (float)row[j] * x[j];
            }
            xout[i] = sum * scales[i];
        }

        // Wait for Core 0 to complete
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    }
};

// ==============================================================================
// Gemma 4 Single Transformer Layer Weights (INT16 Format)
// ==============================================================================
struct GemmaInt16LayerWeights {
    const float* attn_norm;     // [GEMMA_HIDDEN_DIM]
    GemmaQ16Matrix wq;          // [GEMMA_HIDDEN_DIM, GEMMA_HIDDEN_DIM]
    GemmaQ16Matrix wk;          // [GEMMA_HIDDEN_DIM, GEMMA_HIDDEN_DIM]
    GemmaQ16Matrix wv;          // [GEMMA_HIDDEN_DIM, GEMMA_HIDDEN_DIM]
    GemmaQ16Matrix wo;          // [GEMMA_HIDDEN_DIM, GEMMA_HIDDEN_DIM]
    const float* ffn_norm;      // [GEMMA_HIDDEN_DIM]
    GemmaQ16Matrix w_gate;      // [GEMMA_INTERMEDIATE_DIM, GEMMA_HIDDEN_DIM]
    GemmaQ16Matrix w_up;        // [GEMMA_INTERMEDIATE_DIM, GEMMA_HIDDEN_DIM]
    GemmaQ16Matrix w_down;      // [GEMMA_HIDDEN_DIM, GEMMA_INTERMEDIATE_DIM]

    static GemmaInt16LayerWeights from_buffer(const uint8_t* base, uint32_t& offset) {
        GemmaInt16LayerWeights l;
        l.attn_norm = (const float*)(base + offset);
        offset += GEMMA_HIDDEN_DIM * sizeof(float);

        l.wq = GemmaQ16Matrix::from_buffer(base, GEMMA_HIDDEN_DIM, GEMMA_HIDDEN_DIM, offset);
        l.wk = GemmaQ16Matrix::from_buffer(base, GEMMA_HIDDEN_DIM, GEMMA_HIDDEN_DIM, offset);
        l.wv = GemmaQ16Matrix::from_buffer(base, GEMMA_HIDDEN_DIM, GEMMA_HIDDEN_DIM, offset);
        l.wo = GemmaQ16Matrix::from_buffer(base, GEMMA_HIDDEN_DIM, GEMMA_HIDDEN_DIM, offset);

        l.ffn_norm = (const float*)(base + offset);
        offset += GEMMA_HIDDEN_DIM * sizeof(float);

        l.w_gate = GemmaQ16Matrix::from_buffer(base, GEMMA_INTERMEDIATE_DIM, GEMMA_HIDDEN_DIM, offset);
        l.w_up   = GemmaQ16Matrix::from_buffer(base, GEMMA_INTERMEDIATE_DIM, GEMMA_HIDDEN_DIM, offset);
        l.w_down = GemmaQ16Matrix::from_buffer(base, GEMMA_HIDDEN_DIM, GEMMA_INTERMEDIATE_DIM, offset);

        return l;
    }
};

// ==============================================================================
// Tied Token Table (INT16): input embeddings AND output classifier
// ==============================================================================
// Row t is token t's embedding. The same rows, used as a [VOCAB x HIDDEN] matrix,
// produce the logits, so the model stores the table once (as Gemma does).
struct GemmaQuantizedTableQ16 {
    const float* scales;       // [GEMMA_VOCAB_SIZE] float32 scales
    const int16_t* weights;    // [GEMMA_VOCAB_SIZE, GEMMA_HIDDEN_DIM] int16 weights

    static GemmaQuantizedTableQ16 from_buffer(const uint8_t* base, uint32_t& offset) {
        GemmaQuantizedTableQ16 q;
        q.scales = (const float*)(base + offset);
        offset += GEMMA_VOCAB_SIZE * sizeof(float);
        q.weights = (const int16_t*)(base + offset);
        offset += GEMMA_VOCAB_SIZE * GEMMA_HIDDEN_DIM * sizeof(int16_t);
        return q;
    }

    // The table viewed as the output classifier: logits = table . x (no embedding scale)
    inline GemmaQ16Matrix as_classifier() const {
        GemmaQ16Matrix m;
        m.out_dim = GEMMA_VOCAB_SIZE;
        m.in_dim = GEMMA_HIDDEN_DIM;
        m.scales = scales;
        m.weights = weights;
        return m;
    }

    // Embeddings dequantize + Gemma sqrt(d_model) scaling
    inline void dequantize_embedding(uint32_t token_id, float* out_vec) const {
        if (token_id >= GEMMA_VOCAB_SIZE) token_id = 0;
        float s = scales[token_id] * GEMMA_EMBED_SCALE;
        const int16_t* row = &weights[token_id * GEMMA_HIDDEN_DIM];
        for (int i = 0; i < GEMMA_HIDDEN_DIM; i++) {
            out_vec[i] = (float)row[i] * s;
        }
    }
};

// ==============================================================================
// Gemma 4 Core Mathematics
// ==============================================================================

// Gemma RMSNorm with signature (1.0 + weight[i]) unit offset
static inline void gemma_rmsnorm(float* out, const float* x, const float* weight, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf((sum / (float)size) + 1e-6f);
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * scale * (1.0f + weight[i]);
    }
}

// Gemma GeGLU Activation: gelu(gate) * up
// Fast sigmoid approximation: gelu(x) = x / (1.0 + exp(-1.702 * x))
static inline void gemma_geglu(float* out, const float* gate, const float* up, int size) {
    for (int i = 0; i < size; i++) {
        float g = gate[i];
        float gelu = g / (1.0f + expf(-1.702f * g));
        out[i] = gelu * up[i];
    }
}

// Rotary Positional Embeddings (RoPE)
static inline void gemma_apply_rope(float* q_vec, float* k_vec, int pos) {
    for (int h = 0; h < GEMMA_NUM_HEADS; h++) {
        float* q_head = &q_vec[h * GEMMA_HEAD_DIM];
        float* k_head = &k_vec[h * GEMMA_HEAD_DIM];
        for (int i = 0; i < GEMMA_HEAD_DIM; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / (float)GEMMA_HEAD_DIM);
            float theta = (float)pos * freq;
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);

            float q0 = q_head[i], q1 = q_head[i + 1];
            q_head[i]     = q0 * cos_theta - q1 * sin_theta;
            q_head[i + 1] = q0 * sin_theta + q1 * cos_theta;

            float k0 = k_head[i], k1 = k_head[i + 1];
            k_head[i]     = k0 * cos_theta - k1 * sin_theta;
            k_head[i + 1] = k0 * sin_theta + k1 * cos_theta;
        }
    }
}
