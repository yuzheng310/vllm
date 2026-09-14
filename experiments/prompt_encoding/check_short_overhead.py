# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-target short-input control; its times are never counted as project gains."""

import argparse
import json
import statistics
import time
from pathlib import Path

from bench_cpu import Cache, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--backend", choices=["hf", "fastokens"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.backend == "fastokens":
        import fastokens

        fastokens.patch_transformers()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    cache = Cache.create(tok, max_bytes=16 * 1024 * 1024)
    assert cache is not None
    # Predeclared overhead control: first request's first 256 characters, 1000
    # warm calls per arm. This is not another sampled performance workload.
    text = json.loads(args.prepared.read_text())["requests"][0]["text"][:256]
    expected = tok(text, add_special_tokens=False)["input_ids"]
    assert cache.encode(text, add_special_tokens=False) is None
    rounds = []
    for enabled in (False, True, True, False):
        values = []
        for _ in range(1000):
            start = time.perf_counter_ns()
            result = cache.encode(text, add_special_tokens=False) if enabled else None
            if result is None:
                result = tok(text, add_special_tokens=False)["input_ids"]
            values.append(time.perf_counter_ns() - start)
            assert result == expected
        rounds.append(
            dict(
                cache_enabled=enabled,
                mean_us=statistics.mean(values) / 1000,
                median_us=statistics.median(values) / 1000,
                durations_ns=values,
            )
        )
    native = statistics.mean(r["mean_us"] for r in rounds if not r["cache_enabled"])
    candidate = statistics.mean(r["mean_us"] for r in rounds if r["cache_enabled"])
    result = dict(
        scope="256-character fallback overhead control, not serving or a gain claim",
        backend=args.backend,
        prepared_sha256=digest(args.prepared),
        native_mean_us=native,
        cache_mean_us=candidate,
        added_mean_us=candidate - native,
        cache=cache.snapshot(),
        rounds=rounds,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rounds"}))


if __name__ == "__main__":
    main()
