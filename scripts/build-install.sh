#!/bin/bash

set -e

# Remove installation target directory
rm -rf install

cmake --fresh -S . -B build-install -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DLLAMA_TESTS_INSTALL=OFF \
  -DCMAKE_INSTALL_PREFIX="${PWD}/install" \
  -DGGML_BACKEND_DIR="${PWD}/install/lib"
cmake --build build-install --parallel 12
cmake --install build-install

# Test:
#strings install/lib/libggml.so | grep "install/lib"
#env LD_LIBRARY_PATH=$PWD/install/lib ./install/bin/llama-completion -m models/granite-3.1-1b-a400m-instruct.gguf -p "Hello" -n 10 -no-cnv --no-warmup
