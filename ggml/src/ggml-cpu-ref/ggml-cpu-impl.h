#pragma once

// GGML CPU internal header

#include "ggml.h"
#include "ggml-impl.h"

#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#include <stdbool.h>
#include <string.h> // memcpy
#include <math.h>   // fabsf

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;
};


#if defined(_MSC_VER)

#define m512bh(p) p
#define m512i(p) p

#else

#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)

#endif


// TODO: move to ggml-threading
void ggml_barrier(struct ggml_threadpool * tp);

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value);
int  ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value);

#ifdef __cplusplus
}
#endif
