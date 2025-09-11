#pragma once

#include "ggml-cpu-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// "simd mappings" for cpu-ref backend, so only the generic C fallbacks are used
//

// FP16 to FP32 conversion - generic fallback only

// For the cpu-ref backend, we use only generic fallback implementations.

// precomputed f32 table for f16 (256 KB)
// defined in ggml-cpu.c, initialized in ggml_cpu_init()
extern float ggml_table_f32_f16[1 << 16];

// Generic fallback FP16/FP32 conversion using lookup table
inline static float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    return ggml_table_f32_f16[s];
}

#define GGML_CPU_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
#define GGML_CPU_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

#ifdef __cplusplus
}
#endif
