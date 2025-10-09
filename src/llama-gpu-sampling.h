#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_sampler * llama_sampler_gpu_init_greedy(void);

struct llama_sampler * llama_sampler_gpu_init_temp(float temp);

struct llama_sampler * llama_sampler_gpu_init_softmax(void);

struct llama_sampler * llama_sampler_gpu_init_top_k(int32_t k);

#ifdef __cplusplus
}
#endif
