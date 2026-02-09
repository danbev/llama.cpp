#!/bin/bash

get_build_dir() {
    local preset_name=$1
    local llama_dir=${LLAMA_CPP_DIR:-"/home/danbev/work/llama.cpp"}
    echo "$llama_dir/build-$preset_name"
}

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    if [ -z "$LLAMA_PRESET_PREFIX" ]; then
        echo "Error: LLAMA_PRESET_PREFIX not set" >&2
        return 1
    fi
    
    export LLAMA_DEBUG_BUILD=$(get_build_dir "${LLAMA_PRESET_PREFIX}-debug")
    export LLAMA_RELEASE_BUILD=$(get_build_dir "${LLAMA_PRESET_PREFIX}-release")
    echo "Debug build: $LLAMA_DEBUG_BUILD" >&2
    echo "Release build: $LLAMA_RELEASE_BUILD" >&2
else
    get_build_dir "metal-debug"
    get_build_dir "metal-release"
fi
