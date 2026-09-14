# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay the frozen chat inputs through the actual HTTP render endpoint."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import httpx
from bench_cpu import digest, percentile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = json.loads(args.prepared.read_text())
    records = []
    with httpx.Client(base_url=args.url, timeout=60, trust_env=False) as client:
        assert client.get("/health").status_code == 200
        for request in frozen["requests"]:
            body = dict(
                model="repocompass",
                messages=request["messages"],
                tools=request["tools"],
                max_tokens=1,
                add_special_tokens=False,
            )
            start = time.perf_counter_ns()
            response = client.post("/v1/chat/completions/render", json=body)
            elapsed = time.perf_counter_ns() - start
            assert response.status_code == 200, response.text[:1000]
            actual = response.json()["token_ids"]
            assert actual == request["input_ids"], (
                request["task_id"],
                request["turn_id"],
                len(actual),
                len(request["input_ids"]),
            )
            records.append(
                dict(
                    task_id=request["task_id"],
                    turn_id=request["turn_id"],
                    start_ns=start,
                    elapsed_ns=elapsed,
                    text_sha256=hashlib.sha256(request["text"].encode()).hexdigest(),
                    prompt_tokens=len(actual),
                )
            )
        # Correctness-only checks after the timed cold-to-warm replay.
        request = frozen["requests"][-1]
        body = dict(
            model="repocompass",
            messages=request["messages"],
            tools=request["tools"],
            max_tokens=1,
            add_special_tokens=False,
        )
        offsets = client.post(
            "/v1/chat/completions/render", json={**body, "return_token_offsets": True}
        )
        if os.environ.get("VLLM_USE_FASTOKENS") == "1":
            assert offsets.status_code == 501, offsets.text[:1000]
            assert offsets.json()["error"]["message"] == (
                "fastokens does not track character offsets"
            )
            offset_status = "native-unsupported-501"
        else:
            assert offsets.status_code == 200, offsets.text[:1000]
            data = offsets.json()
            assert data["token_ids"] == request["input_ids"]
            assert len(data["token_offsets"]) == len(data["token_ids"])
            offset_status = "PASS"
        truncations = []
        for side in ("left", "right"):
            result = client.post(
                "/v1/chat/completions/render",
                json={**body, "truncate_prompt_tokens": 128, "truncation_side": side},
            )
            assert result.status_code == 200, result.text[:1000]
            expected = (
                request["input_ids"][-128:]
                if side == "left"
                else request["input_ids"][:128]
            )
            assert result.json()["token_ids"] == expected
            truncations.append(side)

        def concurrent_request(index):
            item = frozen["requests"][-8 + index]
            response = client.post(
                "/v1/chat/completions/render",
                json=dict(
                    model="repocompass",
                    messages=item["messages"],
                    tools=item["tools"],
                    max_tokens=1,
                    add_special_tokens=False,
                    cache_salt="thread-check",
                ),
            )
            assert response.status_code == 200, response.text[:1000]
            assert response.json()["token_ids"] == item["input_ids"]
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            concurrent_checks = list(pool.map(concurrent_request, range(8)))
    values = [r["elapsed_ns"] / 1e6 for r in records]
    summary = dict(
        total_http_ms=sum(values),
        p50_http_ms=statistics.median(values),
        p95_http_ms=percentile(values, 0.95),
        exact_requests=len(records),
        offsets=offset_status,
        truncations=truncations,
        concurrent_exact_requests=sum(concurrent_checks),
    )
    args.output.write_text(
        json.dumps(
            dict(
                scope="Loopback HTTP chat rendering only; sequential fixed interleave",
                prepared_sha256=digest(args.prepared),
                summary=summary,
                requests=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
