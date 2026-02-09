#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/get-build-dir.sh

CMD="llama-cli"
USE_DEBUG=false
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug|-d)
            USE_DEBUG=true
            shift
            ;;
        --model|-m)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ "$USE_DEBUG" = true ]; then
    build_dir=$LLAMA_DEBUG_BUILD
else
    build_dir=$LLAMA_RELEASE_BUILD
fi

cmake --build ${build_dir} -j12 --target ${CMD}


# Determine model path: use -m option, then LLAMA_MODEL env var, then default
if [ -z "$MODEL_PATH" ]; then
    if [ -n "$LLAMA_MODEL" ]; then
        model="$LLAMA_MODEL"
    fi
else
    model="$MODEL_PATH"
fi

CMD_OPTIONS=(
    -m "${model}"
    --no-warmup
    -p "What is the capital of Sweden?"
    -n 10
    --verbose-prompt
)

echo "Running ${CMD} with options: ${CMD_OPTIONS[@]}"

if [ "$USE_DEBUG" = true ]; then
    if [[ "$LLAMA_PRESET_PREFIX" == "metal" ]]; then
        lldb ${build_dir}/bin/${CMD} -- "${CMD_OPTIONS[@]}"
    else
        gdb --args ${build_dir}/bin/${CMD} "${CMD_OPTIONS[@]}"
    fi
else
    ${build_dir}/bin/${CMD} "${CMD_OPTIONS[@]}"
fi
