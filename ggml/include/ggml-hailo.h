#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t ggml_backend_hailo_init(void);

GGML_BACKEND_API bool ggml_backend_is_hailo(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hailo_reg(void);

#ifdef  __cplusplus
}
#endif
