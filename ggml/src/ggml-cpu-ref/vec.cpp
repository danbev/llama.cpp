#include "vec.h"

#include <cassert>

// precomputed gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc) {
   assert(nrc == 1);
   GGML_UNUSED(nrc);
   GGML_UNUSED(bx);
   GGML_UNUSED(by);
   GGML_UNUSED(bs);

    // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }

    *s = sumf;
}

void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    int i = 0;
    ggml_float sumf = 0;

    for (; i < n; ++i) {
        sumf += (ggml_float)(GGML_BF16_TO_FP32(x[i]) *
                             GGML_BF16_TO_FP32(y[i]));
    }
    *s = sumf;
}

void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    ggml_float sumf = 0.0;

    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[i])*GGML_CPU_FP16_TO_FP32(y[i]));
    }

    *s = sumf;
}

void ggml_vec_silu_f32(const int n, float * y, const float * x) {
    int i = 0;
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}

void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g) {
    int i = 0;
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]) * g[i];
    }
}

ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    ggml_float sum = 0;
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (ggml_float)val;
        y[i] = val;
    }
    return sum;
}

ggml_float ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max) {
    // log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) - log(soft_max_i)

    int i = 0;
    ggml_float sum = 0;
    for (; i < n; ++i) {
        float val = x[i] - max;
        y[i] = val;
        sum += (ggml_float)expf(val);
    }
    return sum = (ggml_float)logf(sum);
}
