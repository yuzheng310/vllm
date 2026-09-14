# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Match real renderer calls to HTTP requests and compare fixed paired rounds."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from bench_cpu import digest, percentile


def summarize(directory, backend, label):
    http_path = directory / f"{backend}_{label}.http.json"
    renderer_path = directory / f"{backend}_{label}.renderer.json"
    http = json.loads(http_path.read_text())
    renderer = json.loads(renderer_path.read_text())
    rows = []
    first_tasks = set()
    for request in http["requests"]:
        matches = [
            call
            for call in renderer["requests"]
            if request["start_ns"]
            <= call["start_ns"]
            <= request["start_ns"] + request["elapsed_ns"]
            and call["text_sha256"] == request["text_sha256"]
        ]
        assert len(matches) == 1, (backend, label, request["task_id"])
        call = matches[0]
        assert call["tokens"] == request["prompt_tokens"]
        rows.append(
            dict(
                task_id=request["task_id"],
                turn_id=request["turn_id"],
                stage_ms=call["elapsed_ns"] / 1e6,
                cpu_ms=call["cpu_ns"] / 1e6,
                http_ms=request["elapsed_ns"] / 1e6,
                first_turn=request["task_id"] not in first_tasks,
            )
        )
        first_tasks.add(request["task_id"])
    assert len(rows) == 233
    stage = [row["stage_ms"] for row in rows]
    http_times = [row["http_ms"] for row in rows]
    summary = dict(
        backend=backend,
        round=label,
        stage_ms=sum(stage),
        http_ms=sum(http_times),
        stage_cpu_ms=sum(row["cpu_ms"] for row in rows),
        stage_p50_ms=statistics.median(stage),
        stage_p95_ms=percentile(stage, 0.95),
        http_p50_ms=statistics.median(http_times),
        http_p95_ms=percentile(http_times, 0.95),
        first_request_stage_ms=stage[0],
        first_request_http_ms=http_times[0],
        first_turn_stage_ms=sum(row["stage_ms"] for row in rows if row["first_turn"]),
        exact_requests=http["summary"]["exact_requests"],
        cache=renderer["cache"],
        backend_class=renderer["backend"],
        base_sha=renderer["commit"],
        diff_sha=renderer["diff_sha256"],
        code_sha=renderer["code_sha256"],
        versions=renderer["versions"],
        environment=renderer["environment"],
        http_sha=digest(http_path),
        renderer_sha=digest(renderer_path),
        prepared_sha=http["prepared_sha256"],
    )
    assert http["summary"]["concurrent_exact_requests"] == 8
    assert http["summary"]["offsets"] == (
        "PASS" if backend == "hf" else "native-unsupported-501"
    )
    assert http["summary"]["truncations"] == ["left", "right"]
    assert sum(
        call["cache_salt"] == "thread-check" for call in renderer["requests"]
    ) == (8 if label.startswith("b") else 0)
    return dict(summary=summary, requests=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rounds = [
        summarize(args.directory, backend, label)
        for backend in ("hf", "fastokens")
        for label in ("a1", "b1", "b2", "a2")
    ]
    reference = rounds[0]["summary"]
    for result in rounds:
        for key in ("base_sha", "code_sha", "versions", "prepared_sha"):
            assert result["summary"][key] == reference[key], key
    comparisons = {}
    for backend in ("hf", "fastokens"):
        group = [r["summary"] for r in rounds if r["summary"]["backend"] == backend]
        assert len({r["diff_sha"] for r in group}) == 1
        native = [r for r in group if r["round"].startswith("a")]
        candidate = [r for r in group if r["round"].startswith("b")]
        comparisons[backend] = {}
        for metric in ("stage_ms", "stage_cpu_ms", "http_ms", "first_turn_stage_ms"):
            a = statistics.median(r[metric] for r in native)
            b = statistics.median(r[metric] for r in candidate)
            comparisons[backend][metric] = dict(
                native_median=a,
                cache_median=b,
                reduction_percent=(1 - b / a) * 100,
                pair_reductions_percent=[
                    (1 - cand[metric] / orig[metric]) * 100
                    for orig, cand in zip(native, candidate, strict=True)
                ],
            )
    output = dict(
        scope="Frozen 233-request, sequential loopback HTTP render; no GPU or training",
        comparisons=comparisons,
        rounds=rounds,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
