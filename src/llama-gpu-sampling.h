#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

LLAMA_API struct llama_sampler * llama_sampler_gpu_init_greedy(void);

LLAMA_API struct llama_sampler * llama_sampler_gpu_init_temp(float temp);

LLAMA_API struct llama_sampler * llama_sampler_gpu_init_softmax(void);

LLAMA_API struct llama_sampler * llama_sampler_gpu_init_top_k(int32_t k);

// This is not a real top-p sampler but more like a top-k approximation
// which is the reason this takes a k integer parameter instead of a float
// p parameter.
// TODO: implement real top-p sampling on GPU.
LLAMA_API struct llama_sampler * llama_sampler_gpu_init_top_p(int32_t k);

#ifdef __cplusplus
}
#endif
