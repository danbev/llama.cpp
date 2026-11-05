## Hailo-10H backend
TODO: add introduction/description

### Model conversion
```console
python convert_hf_to_gguf.py models/qwen/Qwen2.5-1.5B-Instruct \
    --hef models/hefs/Qwen2.5-1.5B-Instruct.hef \
    --outfile models/hailo-qwen-2.5.gguf \
    --outtype f16
```
Models can be found here on [Huggingface](https://huggingface.co/collections/danbev/hailo).

### Building
```console
cmake -B build-hailo-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HAILO=ON \
  -DHailoRT_DIR="/path/to/hailort/staging_sdk/lib/cmake/HailoRT" \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_BORINGSSL=ON

cmake --build build-hailo-debug
```

### Running
```console
build/bin/llama-cli -m models/hailo-llama-3.2.gguf \
    --prompt "Hello" \
    --jinja --no-warmup -c 512 -lv 2
```
