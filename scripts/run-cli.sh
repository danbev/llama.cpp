#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/common.sh

CMD="llama-cli"

REMAINING_ARGS=()
parse_common_args "$@"

build_target ${CMD}

model=$(get_model_path)

CMD_OPTIONS=(
    -m "${model}"
    --no-warmup
    -p "What is the capital of Sweden?"
    -n 10
    --verbose-prompt
)

run_command ${CMD} "${CMD_OPTIONS[@]}"
