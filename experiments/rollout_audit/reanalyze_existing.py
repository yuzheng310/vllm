# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recount frozen replay artifacts; perform no inference or timing experiment."""

import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path

WORKLOAD_SHA256 = "e5394db91afad03820a87f7b3ad6ec94339e57e1631951902e35ad7a859c0703"
REPORT_NAMES = (
    "session-release-06b-c16-a1",
    "session-release-06b-c16-a2",
    "session-release-06b-c16-b1",
    "session-release-06b-c16-b2",
    "session-release-06b-c8-fcfs",
    "session-release-06b-c8-affinity",
    "session-invariant-06b-c16-fcfs",
    "session-invariant-06b-c16-affinity",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value):
    return json.dumps(value, separators=(",", ":")).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.workload) != WORKLOAD_SHA256:
        raise ValueError("This analysis requires the previously frozen workload")
    workload = json.loads(args.workload.read_text())
    tasks = {t["task_id"]: t for t in workload["tasks"]}
    previous_lengths = {}
    volume = dict.fromkeys(
        (
            "requests",
            "append_only_continuations",
            "full_token_ids",
            "delta_token_ids",
            "full_field_json_bytes",
            "delta_field_json_bytes",
            "full_field_gzip_bytes",
            "delta_field_gzip_bytes",
        ),
        0,
    )
    for task_id, task in tasks.items():
        previous = []
        for turn, request in enumerate(task["requests"]):
            current = request["input_ids"]
            if current[: len(previous)] != previous:
                raise ValueError(f"Non-append-only input: {task_id}, {turn}")
            previous_lengths[(task_id, turn)] = len(previous)
            full = encode({"prompt_token_ids": current})
            delta = encode(
                {
                    "prompt_token_ids_start": len(previous),
                    "prompt_token_ids": current[len(previous) :],
                }
            )
            reconstructed = previous + json.loads(delta)["prompt_token_ids"]
            if reconstructed != current:
                raise ValueError("Delta reconstruction mismatch")
            volume["requests"] += 1
            volume["append_only_continuations"] += int(bool(previous))
            volume["full_token_ids"] += len(current)
            volume["delta_token_ids"] += len(current) - len(previous)
            volume["full_field_json_bytes"] += len(full)
            volume["delta_field_json_bytes"] += len(delta)
            volume["full_field_gzip_bytes"] += len(gzip.compress(full, mtime=0))
            volume["delta_field_gzip_bytes"] += len(gzip.compress(delta, mtime=0))
            previous = current

    rows = []
    for name in REPORT_NAMES:
        path = args.reports / f"{name}.json"
        report = json.loads(path.read_text())
        if report["status"] != "complete":
            raise ValueError(f"Incomplete report: {path}")
        if report["workload_sha256"] != WORKLOAD_SHA256:
            raise ValueError(f"Different workload: {path}")
        for run in report["runs"]:
            shortfall = affected = later = 0
            requests = []
            for task in run["tasks"]:
                for request in task["requests"]:
                    requests.append(request)
                    turn = request["turn"]
                    if not turn:
                        continue
                    # An aligned previous INPUT prefix, excluding its generation.
                    expected = previous_lengths[(task["task_id"], turn)] // 16 * 16
                    gap = max(0, expected - request["cached_tokens"])
                    shortfall += gap
                    affected += int(gap > 0)
                    later += 1
            if len(requests) != volume["requests"]:
                raise ValueError("Different number of completed requests")
            rows.append(
                {
                    "report": path.name,
                    "report_sha256": digest(path),
                    "source_commit": report["source_commit"],
                    "batch_invariant": report["batch_invariant"],
                    "concurrency": report["concurrency"],
                    "wall_s": run["elapsed_s"],
                    "cache_queries": run["cache_queries"],
                    "cache_hits": run["cache_hits"],
                    "uncached_prompt_tokens": run["cache_queries"] - run["cache_hits"],
                    "later_requests": later,
                    "later_requests_missing_previous_input": affected,
                    "previous_input_block_token_shortfall": shortfall,
                    "median_ttft_s": statistics.median(r["ttft_s"] for r in requests),
                    "preemptions": run["preemptions"],
                }
            )
    result = {
        "workload_sha256": WORKLOAD_SHA256,
        "script_sha256": digest(Path(__file__)),
        "scope": "Offline recount of existing artifacts, no GPU or time measurement",
        "replay_rows": rows,
        "hypothetical_prompt_field_encoding": volume,
        "caveats": [
            "Cache comparison uses 16-token alignment and the previous input only.",
            "Not a global compute lower bound: other shared prefixes may exist.",
            "Historical outputs do not drive the replay's subsequent prompts.",
            "JSON volumes are a proposed field encoding, not measured HTTP traffic.",
            "Delta base identity, digest, fallback and transport headers are excluded.",
            "Gzip is applied separately per field payload, not to full HTTP responses.",
            "No CPU serialization speed or end-to-end gain follows from byte counts.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
