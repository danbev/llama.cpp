# Baseline c-only CPU backend for testing.
ggml_add_backend_library(ggml-cpu-ref)

target_sources(ggml-cpu-ref PRIVATE
    ggml-cpu-ref.cpp
    ggml-cpu-ref.c
    quants.h
    quants.c
    vec.h
    vec.cpp
    ops.h
    ops.cpp
    repack.h
    repack.cpp
    traits.h
    traits.cpp
    binary-ops.h
    binary-ops.cpp
    unary-ops.h
    unary-ops.cpp
    ggml-cpu-impl.h
    common.h
    )

if (GGML_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
        target_compile_definitions(ggml-cpu-ref PRIVATE GGML_USE_OPENMP)

        target_link_libraries(ggml-cpu-ref PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    else()
        set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
        message(WARNING "OpenMP not found")
    endif()
endif()

target_compile_features(ggml-cpu-ref PRIVATE cxx_std_17)
target_include_directories(ggml-cpu-ref PRIVATE . ../include)

# Explicitly disable all optimizations and SIMD to ensure pure C
set_target_properties(ggml-cpu-ref PROPERTIES COMPILE_FLAGS "-O0 -fno-tree-vectorize")
