find_package(HailoRT REQUIRED)

message(STATUS "HailoRT found, building Hailo NPU backend")

ggml_add_backend_library(ggml-hailo ggml-hailo.cpp)

target_link_libraries(ggml-hailo PRIVATE HailoRT::libhailort)
