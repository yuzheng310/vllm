# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure complete incremental scoring with cold first turn and bounded APC."""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from benchmark import PromptDiagnostics, active_compute_pids  # noqa: F401

import vllm
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


def tokenize(tokenizer, trace):
    tokens = tokenizer.encode(trace["system"] + "\nUser: " + trace["task"])
    calls = []
    for turn in trace["turns"]:
        tokens += tokenizer.encode("\nAssistant: ", add_special_tokens=False)
        start = len(tokens)
        tokens += tokenizer.encode(turn["action"], add_special_tokens=False)
        calls.append(
            {"tokens": tokens.copy(), "positions": list(range(start, len(tokens)))}
        )
        if "tool_result" in turn:
            tokens += tokenizer.encode(
                "\nTool: " + turn["tool_result"], add_special_tokens=False
            )
    return {"name": trace["name"], "calls": calls}


def execute(llm, calls, mode):
    outputs, times = [], []
    for call in calls:
        params = SamplingParams(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
            detokenize=False,
            prompt_logprobs=0,
            flat_logprobs=True,
            prompt_logprob_positions=(
                call["positions"] if mode in ("sparse", "cache") else None
            ),
            skip_reading_prefix_cache=mode in ("full", "sparse"),
        )
        start = time.perf_counter()
        output = llm.generate(
            [{"prompt_token_ids": call["tokens"]}], params, use_tqdm=False
        )[0]
        times.append(time.perf_counter() - start)
        outputs.append(output)
    return outputs, times


def extract(calls, outputs, mode):
    values, missing = [], []
    for turn, (call, output) in enumerate(zip(calls, outputs)):
        scores = output.prompt_logprobs
        offset = (output.num_cached_tokens or 0) if mode == "unsafe" else 0
        for position in call["positions"]:
            idx = position - offset
            token = call["tokens"][position]
            if (
                scores is None
                or idx < 1
                or idx >= len(scores)
                or token not in scores[idx]
            ):
                missing.append([turn, position])
                values.append(None)
            else:
                values.append(scores[idx][token].logprob)
    return values, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite results")
    if active_compute_pids():
        parser.error("GPU compute is busy")
    modes = ["full", "sparse", "cache", "unsafe"]
    source_root = Path(vllm.__file__).parent
    report = {
        "status": "running",
        "runs": args.runs,
        "environment": {
            "gpu": current_platform.get_device_name(),
            "model": Path(args.model).name,
            "weights": "downloaded model weights",
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "python": platform.python_version(),
            "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "workload_sha256": hashlib.sha256(args.workload.read_bytes()).hexdigest(),
            "source_sha256": {
                p: hashlib.sha256((source_root / p).read_bytes()).hexdigest()
                for p in [
                    "sampling_params.py",
                    "v1/core/kv_cache_manager.py",
                    "v1/engine/input_processor.py",
                    "v1/engine/logprobs.py",
                    "v1/worker/gpu/sample/prompt_logprob.py",
                ]
            },
        },
        "config": {
            "dtype": "bfloat16",
            "chunk": 512,
            "max_model_len": 16384,
            "batch": 1,
            "apc": True,
            "attention": "FLASH_ATTN",
            "modes": modes,
            "cache_reset_before_each_trace": True,
        },
        "diagnostics": [],
        "measurements": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    save()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=16384,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        attention_backend="FLASH_ATTN",
        gpu_memory_utilization=0.65,
        worker_extension_cls="benchmark.PromptDiagnostics",
    )
    traces = [
        tokenize(llm.get_tokenizer(), t)
        for t in json.loads(args.workload.read_text())["traces"]
    ]
    report["traces"] = traces
    save()
    own = active_compute_pids()
    for trace_index, trace in enumerate(traces):
        calls = trace["calls"]
        reference = None
        eligible = []
        for mode in modes:
            for _ in range(2):
                assert llm.reset_prefix_cache()
                execute(llm, calls, mode)
            assert llm.reset_prefix_cache()
            llm.collective_rpc("start_sparse_diagnostics")
            try:
                outputs, _ = execute(llm, calls, mode)
            finally:
                counters = llm.collective_rpc("stop_sparse_diagnostics")[0]
            values, missing = extract(calls, outputs, mode)
            if mode == "full":
                assert not missing
                reference = values
            paired = [(v, r) for v, r in zip(values, reference) if v is not None]
            diagnostic = {
                "trace": trace_index,
                "mode": mode,
                "scenario": "cold-first-turn",
                "cached_tokens": [o.num_cached_tokens for o in outputs],
                "missing": missing,
                "action_logprobs": values,
                "max_abs_error": max(abs(v - r) for v, r in paired) if paired else None,
                **counters,
            }
            report["diagnostics"].append(diagnostic)
            save()
            if mode != "unsafe":
                assert not missing, diagnostic
                assert diagnostic["max_abs_error"] <= 1e-4, diagnostic["max_abs_error"]
            if not missing and diagnostic["max_abs_error"] <= 1e-4:
                eligible.append(mode)
            # Repeating a scored prefix exercises over-hit without invented data.
            repeated, _ = execute(llm, calls, mode)
            repeated_values, repeated_missing = extract(calls, repeated, mode)
            repeated_error = max(
                (
                    abs(v - r)
                    for v, r in zip(repeated_values, reference)
                    if v is not None
                ),
                default=None,
            )
            report["diagnostics"].append(
                {
                    "trace": trace_index,
                    "mode": mode,
                    "scenario": "repeat-after-complete",
                    "cached_tokens": [o.num_cached_tokens for o in repeated],
                    "missing": repeated_missing,
                    "action_logprobs": repeated_values,
                    "max_abs_error": repeated_error,
                }
            )
            save()
            if mode != "unsafe":
                assert not repeated_missing
                assert repeated_error <= 1e-4, repeated_error
        # Diagnostics first. Formal timing is never collected on missing scores.
        for repetition in range(args.runs):
            assert not active_compute_pids() - own, "Another GPU process appeared"
            order = eligible if repetition % 2 == 0 else eligible[::-1]
            for mode in order:
                assert llm.reset_prefix_cache()
                outputs, seconds = execute(llm, calls, mode)
                _, missing = extract(calls, outputs, mode)
                assert not missing
                report["measurements"].append(
                    {
                        "trace": trace_index,
                        "mode": mode,
                        "repetition": repetition,
                        "seconds": sum(seconds),
                        "turn_seconds": seconds,
                        "cached_tokens": [o.num_cached_tokens for o in outputs],
                    }
                )
                save()
    report["status"] = "complete"
    save()
    for trace_index in range(len(traces)):
        for mode in modes:
            times = [
                r["seconds"]
                for r in report["measurements"]
                if r["trace"] == trace_index and r["mode"] == mode
            ]
            if times:
                print(trace_index, mode, float(np.median(times)), flush=True)


if __name__ == "__main__":
    main()
