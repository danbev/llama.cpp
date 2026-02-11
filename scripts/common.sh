#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/get-build-dir.sh

USE_DEBUG=false
MODEL_PATH=""

# Parse common arguments
parse_common_args() {
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
                # Return unknown args for script-specific handling
                REMAINING_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

# Build the target
build_target() {
    local cmd=$1
    if [ "$USE_DEBUG" = true ]; then
        build_dir=$LLAMA_DEBUG_BUILD
    else
        build_dir=$LLAMA_RELEASE_BUILD
    fi

    cmake --build ${build_dir} -j12 --target ${cmd}
}

# Get the model path
get_model_path() {
    if [ -z "$MODEL_PATH" ]; then
        if [ -n "$LLAMA_MODEL" ]; then
            echo "$LLAMA_MODEL"
        fi
    else
        echo "$MODEL_PATH"
    fi
}

# Run the command with options
run_command() {
    local cmd=$1
    shift
    local cmd_options=("$@")

    echo "Running ${cmd} with options: ${cmd_options[@]}"

    if [ "$USE_DEBUG" = true ]; then
        if [[ "$LLAMA_PRESET_PREFIX" == "metal" ]]; then
            lldb ${build_dir}/bin/${cmd} -- "${cmd_options[@]}"
        else
            gdb --args ${build_dir}/bin/${cmd} "${cmd_options[@]}"
        fi
    else
        ${build_dir}/bin/${cmd} "${cmd_options[@]}"
    fi
}
