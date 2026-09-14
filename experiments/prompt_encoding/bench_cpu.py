# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frozen CPU encoding comparison; never loads model weights or starts a GPU."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Cache = load_module(
    "encoding_cache", ROOT / "vllm/tokenizers/chatml_encoding_cache.py"
).ChatMLEncodingCache


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(workload_path, source, output, include_messages=False):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    workload = json.loads(workload_path.read_text())
    parquet = source / "raw/swe_bench_verified.parquet"
    reconstruction = source / "reconstruct_tokens.py"
    assert digest(parquet) == workload["source_parquet_sha256"]
    assert digest(reconstruction) == workload["reconstruction_sha256"]
    module = load_module("reconstruction", reconstruction)
    wanted = {task["task_id"] for task in workload["tasks"]}
    rows = {
        row["instance_id"]: row
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=32)
        for row in batch.to_pylist()
        if row["instance_id"] in wanted
    }
    tokenizer_path = source / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tasks = []
    for task in workload["tasks"]:
        chat = rows[task["task_id"]]["chat_messages"]
        messages = [
            module._normalize_message(m) for m in module._as_list(chat["messages"])
        ]
        requests = []
        for request, (prefix, _) in zip(
            task["requests"], module.conversation_turns(messages), strict=True
        ):
            text = tokenizer.apply_chat_template(
                prefix,
                tools=module._as_list(chat.get("tools")) or None,
                add_generation_prompt=True,
                tokenize=False,
            )
            assert (
                tokenizer(text, add_special_tokens=False)["input_ids"]
                == request["input_ids"]
            )
            requests.append(
                dict(
                    task_id=task["task_id"],
                    turn_id=request["turn_id"],
                    text=text,
                    input_ids=request["input_ids"],
                )
            )
            if include_messages:
                requests[-1]["messages"] = prefix
                requests[-1]["tools"] = module._as_list(chat.get("tools")) or None
        tasks.append(requests)
    # Eight active trajectories, one request per slot per sweep. Refill a slot
    # on the next sweep. This is a deterministic interleave, not tool timing.
    pending = iter(tasks)
    active = [iter(next(pending)) for _ in range(8)]
    ordered = []
    while active:
        remaining = []
        for trajectory in active:
            request = next(trajectory, None)
            if request is None:
                replacement = next(pending, None)
                if replacement is not None:
                    remaining.append(iter(replacement))
            else:
                ordered.append(request)
                remaining.append(trajectory)
        active = remaining
    result = dict(
        workload_sha256=digest(workload_path),
        source_parquet_sha256=digest(parquet),
        reconstruction_sha256=digest(reconstruction),
        tokenizer_revision=workload["tokenizer_revision"],
        tokenizer_json_sha256=digest(tokenizer_path / "tokenizer.json"),
        chat_template_sha256=digest(tokenizer_path / "chat_template.jinja"),
        schedule="8 active trajectories, round robin, frozen source task order",
        requests=ordered,
    )
    output.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "requests"}))


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (position - low) * (values[high] - values[low])


def measure(args):
    if args.backend == "fastokens":
        import fastokens

        fastokens.patch_transformers()
    from transformers import AutoTokenizer

    frozen = json.loads(args.prepared.read_text())
    assert digest(args.tokenizer / "tokenizer.json") == frozen["tokenizer_json_sha256"]
    # A/B/B/A, plus the ablation twice at fixed positions. Reload the tokenizer
    # each time; no warming on evaluation text before its timed first request.
    rounds = ["native", "cache", "segmented", "segmented", "cache", "native"]
    results = []
    for mode in rounds:
        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        tok("A short unrelated warm-up.", add_special_tokens=False)
        cache = (
            None
            if mode == "native"
            else Cache.create(
                tok, max_bytes=(16 * 1024 * 1024 if mode == "cache" else 0)
            )
        )
        if mode != "native" and cache is None:
            raise ValueError(
                f"Unsupported cache backend: {type(tok.backend_tokenizer)}"
            )
        records = []
        first_tasks = set()
        for request in frozen["requests"]:
            cpu_start = time.process_time_ns()
            start = time.perf_counter_ns()
            ids = (
                tok(request["text"], add_special_tokens=False)["input_ids"]
                if cache is None
                else cache.encode(request["text"], add_special_tokens=False)
            )
            fallback = ids is None
            if fallback:
                ids = tok(request["text"], add_special_tokens=False)["input_ids"]
            elapsed = time.perf_counter_ns() - start
            cpu_elapsed = time.process_time_ns() - cpu_start
            if ids != request["input_ids"]:
                raise ValueError(
                    f"Token mismatch: {args.backend}/{mode}/"
                    f"{request['task_id']}/{request['turn_id']}"
                )
            records.append(
                dict(
                    task_id=request["task_id"],
                    turn_id=request["turn_id"],
                    first_turn=request["task_id"] not in first_tasks,
                    prompt_tokens=len(ids),
                    elapsed_ns=elapsed,
                    cpu_ns=cpu_elapsed,
                    fallback=fallback,
                )
            )
            first_tasks.add(request["task_id"])
        values = [r["elapsed_ns"] / 1e6 for r in records]
        summary = dict(
            mode=mode,
            total_ms=sum(values),
            cpu_ms=sum(r["cpu_ns"] for r in records) / 1e6,
            p50_ms=statistics.median(values),
            p95_ms=percentile(values, 0.95),
            first_turn_ms=sum(r["elapsed_ns"] for r in records if r["first_turn"])
            / 1e6,
            exact_requests=len(records),
            cache=None if cache is None else cache.snapshot(),
        )
        print(json.dumps(summary), flush=True)
        results.append(dict(summary=summary, requests=records))
    args.output.write_text(
        json.dumps(
            dict(
                scope="CPU helper encoding only; not HTTP, GPU or training throughput",
                backend=args.backend,
                backend_class=str(type(tok.backend_tokenizer)),
                prepared_sha256=digest(args.prepared),
                workload={k: v for k, v in frozen.items() if k != "requests"},
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                cache_sha256=digest(ROOT / "vllm/tokenizers/chatml_encoding_cache.py"),
                benchmark_sha256=digest(Path(__file__)),
                platform=platform.platform(),
                processor=platform.processor(),
                python=platform.python_version(),
                versions={
                    name: importlib.metadata.version(name)
                    for name in ("transformers", "tokenizers")
                    + (("fastokens",) if args.backend == "fastokens" else ())
                },
                environment={
                    k: os.environ.get(k)
                    for k in ("TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS")
                },
                rounds=results,
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--include-messages", action="store_true")
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--backend", choices=["hf", "fastokens"], default="hf")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.prepare:
        prepare(args.workload, args.source, args.output, args.include_messages)
    else:
        measure(args)


if __name__ == "__main__":
    main()
