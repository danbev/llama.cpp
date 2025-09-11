// Vectorized functions for fundamental operations

#pragma once

#include "ggml-impl.h"
#include "simd-mappings.h"
#include "ggml.h"
#include "ggml-cpu.h"

// floating point type used to accumulate sums
typedef double ggml_float;

#define GGML_GELU_FP16
#define GGML_GELU_QUICK_FP16

#define GGML_SOFT_MAX_UNROLL 4
#define GGML_VEC_DOT_UNROLL  2
#define GGML_VEC_MAD_UNROLL  32

#ifdef __cplusplus
extern "C" {
#endif

//
// global data
//

// precomputed gelu table for f16 (128 KB)
extern ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
extern ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

//
// fundamental operations
//

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc);
void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc);

void ggml_vec_silu_f32(const int n, float * y, const float * x);
ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max);
ggml_float ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max);

inline static void ggml_vec_set_i8(const int n, int8_t * x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_i16(const int n, int16_t * x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_set_i32(const int n, int32_t * x, const int32_t   v) { for (int i = 0; i < n; ++i) x[i] = v;    }
inline static void ggml_vec_cpy_i32(const int n, int32_t * y, const int32_t * x) { for (int i = 0; i < n; ++i) y[i] = x[i]; }

inline static void ggml_vec_set_f16(const int n, ggml_fp16_t * x, const ggml_fp16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
inline static void ggml_vec_set_bf16(const int n, ggml_bf16_t * x, const ggml_bf16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }

inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) {
    int i = 0;
    for (; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

inline static void ggml_vec_add_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) + GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_add1_f32(const int n, float * z, const float * x, const float   v) { for (int i = 0; i < n; ++i) z[i]  = x[i] + v;    }
inline static void ggml_vec_acc_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i] += x[i];        }
inline static void ggml_vec_acc1_f32(const int n, float * y, const float   v)                  { for (int i = 0; i < n; ++i) y[i] += v;           }
inline static void ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] - y[i]; }
inline static void ggml_vec_sub_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) - GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_set_f32 (const int n, float * x, const float   v)                  { for (int i = 0; i < n; ++i) x[i]  = v;           }
inline static void ggml_vec_cpy_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = x[i];        }
inline static void ggml_vec_neg_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i]  = -x[i];       }
inline static void ggml_vec_neg_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(-GGML_CPU_FP16_TO_FP32(x[i]));
    }
}

inline static void ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];   }
inline static void ggml_vec_mul_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) * GGML_CPU_FP16_TO_FP32(y[i]));
    }
}
inline static void ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i]/y[i];   }
inline static void ggml_vec_div_f16 (const int n, ggml_fp16_t * z, const ggml_fp16_t * x, const ggml_fp16_t * y) {
    for (int i = 0; i < n; ++i) {
        z[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(x[i]) / GGML_CPU_FP16_TO_FP32(y[i]));
    }
}

// compute GGML_VEC_DOT_UNROLL dot products at once
// xs - x row stride in bytes
inline static void ggml_vec_dot_f16_unroll(const int n, const int xs, float * GGML_RESTRICT s, void * GGML_RESTRICT xv, ggml_fp16_t * GGML_RESTRICT y) {
    ggml_float sumf[GGML_VEC_DOT_UNROLL] = { 0.0 };

    ggml_fp16_t * GGML_RESTRICT x[GGML_VEC_DOT_UNROLL];

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        x[i] = (ggml_fp16_t *) ((char *) xv + i*xs);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < GGML_VEC_DOT_UNROLL; ++j) {
            sumf[j] += (ggml_float)(GGML_CPU_FP16_TO_FP32(x[j][i])*GGML_CPU_FP16_TO_FP32(y[i]));
        }
    }

    for (int i = 0; i < GGML_VEC_DOT_UNROLL; ++i) {
        s[i] = (float)sumf[i];
    }
}

inline static void ggml_vec_mad_f32(const int n, float * GGML_RESTRICT y, const float * GGML_RESTRICT x, const float v) {
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] += x[i]*v;
    }
}

inline static void ggml_vec_mad_f16(const int n, ggml_fp16_t * GGML_RESTRICT y, const ggml_fp16_t * GGML_RESTRICT x, const float v) {
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i]) + GGML_CPU_FP16_TO_FP32(x[i])*v);
    }
}

// xs and vs are byte strides of x and v
inline static void ggml_vec_mad_f32_unroll(const int n, const int xs, const int vs, float * GGML_RESTRICT y, const float * GGML_RESTRICT xv, const float * GGML_RESTRICT vv) {

    const float * GGML_RESTRICT x[GGML_VEC_MAD_UNROLL];
    const float * GGML_RESTRICT v[GGML_VEC_MAD_UNROLL];

    for (int i = 0; i < GGML_VEC_MAD_UNROLL; ++i) {
        x[i] = (const float *) ((const char *) xv + i*xs);
        v[i] = (const float *) ((const char *) vv + i*vs);
    }

    // scalar
    for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
        for (int i = 0; i < n; ++i) {
            y[i] += x[k][i]*v[k][0];
        }
    }
}

inline static void ggml_vec_mad1_f32(const int n, float * y, const float * x, const float s, const float b) {
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = x[i]*s + b;
    }
}

inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) {
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
}

inline static void ggml_vec_scale_f16(const int n, ggml_fp16_t * y, const float v) {
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(y[i])*v);
    }
}

inline static void ggml_vec_norm_f32 (const int n, float * s, const float * x) { ggml_vec_dot_f32(n, s, 0, x, 0, x, 0, 1); *s = sqrtf(*s);   }
inline static void ggml_vec_sqr_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i]*x[i];   }
inline static void ggml_vec_sqr_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(v*v);
    }
}
inline static void ggml_vec_sqrt_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sqrtf(x[i]); }
inline static void ggml_vec_sqrt_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(sqrtf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_log_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = logf(x[i]);  }
inline static void ggml_vec_log_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(logf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_sin_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = sinf(x[i]);  }
inline static void ggml_vec_sin_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(sinf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_cos_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = cosf(x[i]);  }
inline static void ggml_vec_cos_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(cosf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_abs_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fabsf(x[i]); }
inline static void ggml_vec_abs_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(fabsf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_sgn_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : ((x[i] < 0.f) ? -1.f : 0.f); }
inline static void ggml_vec_sgn_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? 1.f : ((v < 0.f) ? -1.f : 0.f));
    }
}
inline static void ggml_vec_step_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? 1.f : 0.f; }
inline static void ggml_vec_step_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16((GGML_CPU_FP16_TO_FP32(x[i]) > 0.f) ? 1.f : 0.f);
    }
}
inline static void ggml_vec_tanh_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = tanhf(x[i]);  }
inline static void ggml_vec_tanh_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(tanhf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_elu_f32  (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : expm1f(x[i]); }
inline static void ggml_vec_elu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(expm1f(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}
inline static void ggml_vec_relu_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = (x[i] > 0.f) ? x[i] : 0.f; }
inline static void ggml_vec_relu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? v : 0.f);
    }
}
inline static void ggml_vec_leaky_relu_f32 (const int n, float * y, const float * x, const float ns) { for (int i = 0; i < n; ++i) y[i] = ((x[i] > 0.f) ? x[i] : 0.f) + ns * ((x[i] < 0.0f) ? x[i] : 0.f); }
inline static void ggml_vec_leaky_relu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const float ns) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(((v > 0.f) ? v : 0.f) + ns * ((v < 0.0f) ? v : 0.f));
    }
}
inline static void ggml_vec_sigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = 1.f / (1.f + expf(-x[i])); }
inline static void ggml_vec_sigmoid_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(1.f / (1.f + expf(-GGML_CPU_FP16_TO_FP32(x[i]))));
    }
}
// TODO: optimize performance
inline static void ggml_vec_hardswish_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = x[i] * fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
inline static void ggml_vec_hardswish_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(v * fminf(1.0f, fmaxf(0.0f, (v + 3.0f) / 6.0f)));
    }
}
inline static void ggml_vec_hardsigmoid_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f)); }
inline static void ggml_vec_hardsigmoid_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(fminf(1.0f, fmaxf(0.0f, (GGML_CPU_FP16_TO_FP32(x[i]) + 3.0f) / 6.0f)));
    }
}
inline static void ggml_vec_exp_f32 (const int n, float * y, const float * x) { for (int i = 0; i < n; ++i) y[i] = expf(x[i]); }
inline static void ggml_vec_exp_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(expf(GGML_CPU_FP16_TO_FP32(x[i])));
    }
}

static const float GELU_COEF_A     = 0.044715f;
static const float GELU_QUICK_COEF = -1.702f;
static const float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
static const float SQRT_2_INV      = 0.70710678118654752440084436210484f;

inline static float ggml_gelu_f32(float x) {
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

inline static void ggml_vec_gelu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_table_gelu_f16[i16[i]];
    }
}

inline static void ggml_vec_gelu_erf_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float xi = GGML_CPU_FP16_TO_FP32(x[i]);
        float res = 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV));
        y[i] = GGML_CPU_FP32_TO_FP16(res);
    }
}

#ifdef GGML_GELU_FP16
inline static void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) {
            y[i] = 0.0f;
        } else if (x[i] >= 10.0f) {
            y[i] = x[i];
        } else {
            ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, sizeof(uint16_t));
            y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[t]);
        }
    }
}
#else
inline static void ggml_vec_gelu_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_f32(x[i]);
    }
}
#endif

inline static void ggml_vec_gelu_erf_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        y[i] = 0.5f*xi*(1.0f + erff(xi*SQRT_2_INV));
    }
}

inline static float ggml_gelu_quick_f32(float x) {
    return x*(1.0f/(1.0f+expf(GELU_QUICK_COEF*x)));
}

//inline static void ggml_vec_gelu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
//    const uint16_t * i16 = (const uint16_t *) x;
//    for (int i = 0; i < n; ++i) {
//        y[i] = ggml_table_gelu_quick_f16[i16[i]];
//    }
//}

#ifdef GGML_GELU_QUICK_FP16
inline static void ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[t]);
    }
}
#else
inline static void ggml_vec_gelu_quick_f32(const int n, float * y, const float * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_quick_f32(x[i]);
    }
}
#endif

inline static void ggml_vec_gelu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(v*(1.0f/(1.0f+expf(GELU_QUICK_COEF*v))));
    }
}

// Sigmoid Linear Unit (SiLU) function
inline static float ggml_silu_f32(float x) {
    return x/(1.0f + expf(-x));
}
inline static ggml_fp16_t ggml_silu_f16(ggml_fp16_t x) {
    float v = GGML_CPU_FP16_TO_FP32(x);
    return GGML_CPU_FP32_TO_FP16(v/(1.0f + expf(-v)));
}

#if __FINITE_MATH_ONLY__
#error "some routines in ggml.c require non-finite math arithmetics -- pass -fno-finite-math-only to the compiler to fix"
#error "ref: https://github.com/ggml-org/llama.cpp/pull/7154#issuecomment-2143844461"
#endif

inline static void ggml_vec_silu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_silu_f16(x[i]);
    }
}

inline static float ggml_silu_backward_f32(float x, float dy) {
    const float s = 1.0f/(1.0f + expf(-x));
    return dy*s*(1.0f + x*(1.0f - s));
}

inline static ggml_fp16_t ggml_silu_backward_f16(ggml_fp16_t x, ggml_fp16_t dy) {
    const float v = GGML_CPU_FP16_TO_FP32(x);
    const float s = 1.0f/(1.0f + expf(-v));
    return GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(dy)*s*(1.0f + v*(1.0f - s)));
}

inline static void ggml_vec_silu_backward_f32(const int n, float * dx, const float * x, const float * dy) {
    for (int i = 0; i < n; ++i) {
        dx[i] = ggml_silu_backward_f32(x[i], dy[i]);
    }
}

inline static void ggml_vec_silu_backward_f16(const int n, ggml_fp16_t * dx, const ggml_fp16_t * x, const ggml_fp16_t * dy) {
    for (int i = 0; i < n; ++i) {
        dx[i] = ggml_silu_backward_f16(x[i], dy[i]);
    }
}

inline static void ggml_vec_reglu_f32 (const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        y[i] = (x[i] > 0.f) ? x[i] * g[i] : 0.f;
    }
}

inline static void ggml_vec_reglu_f16 (const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(x[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((v > 0.f) ? v * GGML_CPU_FP16_TO_FP32(g[i]) : 0.f);
    }
}

#ifdef GGML_GELU_FP16
inline static void ggml_vec_geglu_f32(const int n, float * y, const float * x, const float * g) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        if (x[i] <= -10.0f) {
            y[i] = 0.0f;
        } else if (x[i] >= 10.0f) {
            y[i] = x[i] * g[i];
        } else {
            ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
            memcpy(&t, &fp16, sizeof(uint16_t));
            y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[t]) * g[i];
        }
    }
}
#else
inline static void ggml_vec_geglu_f32(const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_f32(x[i]) * g[i];
    }
}
#endif

inline static void ggml_vec_geglu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(ggml_table_gelu_f16[i16[i]]) * v);
    }
}

void ggml_vec_swiglu_f32(const int n, float * y, const float * x, const float * g);

inline static void ggml_vec_swiglu_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    for (int i = 0; i < n; ++i) {
        float xi = GGML_CPU_FP16_TO_FP32(x[i]);
        float gi = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16((xi/(1.0f + expf(-xi))) * gi);
    }
}

inline static void ggml_vec_geglu_erf_f32(const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        float xi = x[i];
        y[i] = 0.5f * xi * (1.0f + erff(xi*SQRT_2_INV)) * g[i];
    }
}

inline static void ggml_vec_geglu_erf_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    for (int i = 0; i < n; ++i) {
        float xi = GGML_CPU_FP16_TO_FP32(x[i]);
        float gi = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(0.5f * xi * (1.0f + erff(xi*SQRT_2_INV)) * gi);
    }
}

#ifdef GGML_GELU_QUICK_FP16
inline static void ggml_vec_geglu_quick_f32(const int n, float * y, const float * x, const float * g) {
    uint16_t t;
    for (int i = 0; i < n; ++i) {
        ggml_fp16_t fp16 = GGML_CPU_FP32_TO_FP16(x[i]);
        memcpy(&t, &fp16, sizeof(uint16_t));
        y[i] = GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[t]) * g[i];
    }
}
#else
inline static void ggml_vec_geglu_quick_f32(const int n, float * y, const float * x, const float * g) {
    for (int i = 0; i < n; ++i) {
        y[i] = ggml_gelu_quick_f32(x[i]) * g[i];
    }
}
#endif

inline static void ggml_vec_geglu_quick_f16(const int n, ggml_fp16_t * y, const ggml_fp16_t * x, const ggml_fp16_t * g) {
    const uint16_t * i16 = (const uint16_t *) x;
    for (int i = 0; i < n; ++i) {
        float v = GGML_CPU_FP16_TO_FP32(g[i]);
        y[i] = GGML_CPU_FP32_TO_FP16(GGML_CPU_FP16_TO_FP32(ggml_table_gelu_quick_f16[i16[i]]) * v);
    }
}

inline static void ggml_vec_sum_f32(const int n, float * s, const float * x) {
    ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (ggml_float)x[i];
    }
    *s = (float)sum;
}

inline static void ggml_vec_sum_f32_ggf(const int n, ggml_float * s, const float * x) {
    ggml_float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (ggml_float)x[i];
    }
    *s = sum;
}

inline static void ggml_vec_sum_f16_ggf(const int n, float * s, const ggml_fp16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += GGML_CPU_FP16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void ggml_vec_sum_bf16_ggf(const int n, float * s, const ggml_bf16_t * x) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += GGML_BF16_TO_FP32(x[i]);
    }
    *s = sum;
}

inline static void ggml_vec_max_f32(const int n, float * s, const float * x) {
    float max = -INFINITY;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
    }
    *s = max;
}

inline static void ggml_vec_norm_inv_f32(const int n, float * s, const float * x) {
    ggml_vec_norm_f32(n, s, x);
    *s = 1.f/(*s);
}

inline static void ggml_vec_argmax_f32(const int n, int * s, const float * x) {
    float max = -INFINITY;
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        max = MAX(max, x[i]);
        if (max == x[i]) { idx = i; }
    }
    *s = idx;
}

#ifdef __cplusplus
}
#endif
