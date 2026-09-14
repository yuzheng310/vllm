# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze a deterministic public code-localization replay before benchmarking."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=48)
    parser.add_argument("--max-model-len", type=int, default=40960)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a frozen workload")
    source = args.source.resolve()
    reconstruction = source / "reconstruct_tokens.py"
    spec = importlib.util.spec_from_file_location("reconstruction", reconstruction)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = AutoTokenizer.from_pretrained(
        source / "tokenizer", local_files_only=True, trust_remote_code=False
    )
    parquet = source / "raw/swe_bench_verified.parquet"
    rows = pq.read_table(parquet).to_pylist()
    rows.sort(key=lambda row: hashlib.sha256(row["instance_id"].encode()).hexdigest())
    tasks, excluded = [], []
    for row in rows:
        task = module.reconstruct_task(tokenizer, row, keep_ids=True)
        reason = None
        for request in task["requests"]:
            if not request["generation_prompt_stable"]:
                reason = "unstable_generation_prefix"
            elif request["recorded_output_tokens"] < 1:
                reason = "missing_recorded_decode_length"
            elif (
                len(request["input_ids"]) + request["recorded_output_tokens"]
                > args.max_model_len
            ):
                reason = "exceeds_model_length"
        if reason is not None or not task["requests"]:
            excluded.append({"task_id": task["task_id"], "reason": reason})
            continue
        tasks.append(task)
        if len(tasks) == args.tasks:
            break
    if len(tasks) != args.tasks:
        raise ValueError("Not enough complete eligible trajectories")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "source_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "reconstruction_sha256": hashlib.sha256(
            reconstruction.read_bytes()
        ).hexdigest(),
        "tokenizer_revision": module.TOKENIZER_REV,
        "selection": "first eligible tasks sorted by sha256(instance_id)",
        "total_source_tasks": len(rows),
        "excluded_before_selection_completed": excluded,
        "max_model_len": args.max_model_len,
        "scope": "fixed historical prompts and recorded decode lengths; not live RL",
        "tasks": tasks,
    }
    args.output.write_text(json.dumps(result))
    print(
        json.dumps(
            {
                "tasks": len(tasks),
                "requests": sum(len(task["requests"]) for task in tasks),
                "excluded": excluded,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
