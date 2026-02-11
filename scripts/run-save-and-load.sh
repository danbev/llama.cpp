#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/common.sh

CMD="llama-save-load-state"

REMAINING_ARGS=()
parse_common_args "$@"

build_target ${CMD}

model=$(get_model_path)

CMD_OPTIONS=(
    -m "${model}" \
    -ngl 10 -c 1024 -fa off --no-op-offload
)

run_command ${CMD} "${CMD_OPTIONS[@]}"
