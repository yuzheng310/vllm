#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# Run after the alternating confirmation, through run-in-workspace.sh.
task_root="$(pwd -P)"
test -f "$task_root/run-in-workspace.sh"
driver="$task_root/src/vllm-session-affinity/experiments/session_affinity/replay.py"
output="$task_root/reports/session-4b-screen-20260914"
phase="${1:?Choose cascade or invariant}"
extra=()
invariant=0
case "$phase" in
    cascade) extra=(--allow-cascade) ;;
    invariant) invariant=1 ;;
    *) exit 2 ;;
esac

for policy in fcfs affinity; do
    name="$policy-c8-$phase"
    source_dir="$task_root/src/vllm-session-baseline"
    if [[ "$policy" == affinity ]]; then
        source_dir="$task_root/src/vllm-session-affinity"
    fi
    test ! -e "$output/$name.json"
    test ! -e "$output/$name.log"
    printf 'Starting %s\n' "$name"
    OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT="$invariant" \
        PYTHONPATH="$source_dir" \
        timeout 1200 "$task_root/.venv/bin/python" "$driver" \
        --model "$task_root/models/Qwen3-4B" \
        --workload "$task_root/experiments/session-affinity-workload.json" \
        --output "$output/$name.json" --policy "$policy" \
        --concurrency 8 --max-wait 1 --runs 1 "${extra[@]}" \
        > "$output/$name.log" 2>&1
    printf 'Completed %s\n' "$name"
done
