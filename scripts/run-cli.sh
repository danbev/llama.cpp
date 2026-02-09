#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/get-build-dir.sh

USE_DEBUG=false
if [[ "$1" == "--debug" || "$1" == "-d" ]]; then
    USE_DEBUG=true
    shift
fi

# Select build directory
if [ "$USE_DEBUG" = true ]; then
    build_dir=$LLAMA_DEBUG_BUILD
else
    build_dir=$LLAMA_RELEASE_BUILD
fi

model=${build_dir}/../models/granite-4.0-h-tiny-Q4_0.gguf
CMD="llama-cli"

# Use array for command options (fixes the quoting issue)
CMD_OPTIONS=(
    -m "${model}"
    --no-warmup
    -p "What is the capital of Sweden?"
    -n 10
    --verbose-prompt
    -no-cnv
)

echo "Running ${CMD} with options: ${CMD_OPTIONS[@]}"

cmake --build ${build_dir} -j12 --target ${CMD}

if [ "$USE_DEBUG" = true ]; then
    if [[ "$LLAMA_PRESET_PREFIX" == "metal" ]]; then
        lldb ${build_dir}/bin/${CMD} -- "${CMD_OPTIONS[@]}"
    else
        gdb --args ${build_dir}/bin/${CMD} "${CMD_OPTIONS[@]}"
    fi
else
    ${build_dir}/bin/${CMD} "${CMD_OPTIONS[@]}"
fi
