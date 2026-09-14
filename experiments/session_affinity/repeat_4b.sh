#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# Invoke through the workspace's run-in-workspace.sh to preserve isolation.
task_root="$(pwd -P)"
test -f "$task_root/run-in-workspace.sh"
driver="$task_root/src/vllm-session-affinity/experiments/session_affinity/replay.py"
output="$task_root/reports/session-4b-screen-20260914"
test -f "$output/affinity-c8-b1.json"

run_case() {
    local name="$1" policy="$2" concurrency="$3" source_dir
    if [[ "$policy" == affinity ]]; then
        source_dir="$task_root/src/vllm-session-affinity"
    else
        source_dir="$task_root/src/vllm-session-baseline"
    fi
    test ! -e "$output/$name.json"
    test ! -e "$output/$name.log"
    printf 'Starting %s\n' "$name"
    OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=0 \
        PYTHONPATH="$source_dir" \
        timeout 1200 "$task_root/.venv/bin/python" "$driver" \
        --model "$task_root/models/Qwen3-4B" \
        --workload "$task_root/experiments/session-affinity-workload.json" \
        --output "$output/$name.json" --policy "$policy" \
        --concurrency "$concurrency" --max-wait 1 --runs 1 \
        > "$output/$name.log" 2>&1
    printf 'Completed %s\n' "$name"
}

# The screen ended with native c4 A1 then candidate B1: finish A/B/B/A.
run_case affinity-c8-b2 affinity 8
run_case fcfs-c4-a2 fcfs 4
