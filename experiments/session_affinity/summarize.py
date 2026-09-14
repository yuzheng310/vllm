# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compatible replay artifacts without selecting favorable runs."""

import argparse
import json
from pathlib import Path

import numpy as np


def describe(report):
    runs = report["runs"]
    requests = [r for run in runs for t in run["tasks"] for r in t["requests"]]
    cold = [r["elapsed_s"] for r in requests if r["turn"] == 0]
    return {
        "runs": len(runs),
        "wall_s": [run["elapsed_s"] for run in runs],
        "median_wall_s": float(np.median([run["elapsed_s"] for run in runs])),
        "trajectories_per_min": [run["trajectories_per_min"] for run in runs],
        "trajectory_active_p95_s": [run["trajectory_p95_s"] for run in runs],
        "batch_completion_p95_s": [
            float(np.percentile([t["completed_s"] for t in run["tasks"]], 95))
            for run in runs
        ],
        "first_turn_p95_s": float(np.percentile(cold, 95)),
        "ttft_p95_s": float(np.percentile([r["ttft_s"] for r in requests], 95)),
        "cached_fraction": sum(run["cache_hits"] for run in runs)
        / max(1, sum(run["cache_queries"] for run in runs)),
        "preemptions": [run["preemptions"] for run in runs],
        "max_waiting": [run["max_waiting"] for run in runs],
        "completed_tasks": [run["completed_tasks"] for run in runs],
        "corrupted_requests": sum(
            bool(r["is_corrupted"])
            for run in runs
            for r in run["finished_request_metrics"]
        ),
    }


def outputs(run):
    return {
        (task["task_id"], request["turn"]): request["output_sha256"]
        for task in run["tasks"]
        for request in task["requests"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline, candidate = [
        json.loads(path.read_text()) for path in (args.baseline, args.candidate)
    ]
    for key in (
        "model",
        "weights_sha256",
        "workload_sha256",
        "script_sha256",
        "concurrency",
        "tool_delay_s",
        "engine_args",
        "batch_invariant",
    ):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Incomparable artifacts: {key}")
    if baseline["status"] != "complete" or candidate["status"] != "complete":
        raise ValueError("Incomplete replay cannot count as performance evidence")
    a, b = describe(baseline), describe(candidate)
    hashes_a = outputs(baseline["runs"][0])
    hashes_b = outputs(candidate["runs"][0])
    if hashes_a.keys() != hashes_b.keys():
        raise ValueError("The compared runs did not complete the same requests")
    result = {
        "baseline": a,
        "candidate": b,
        "median_wall_reduction_percent": 100
        * (1 - b["median_wall_s"] / a["median_wall_s"]),
        "first_run_output_hash_matches": sum(
            hashes_a[key] == hashes_b[key] for key in hashes_a
        ),
        "first_run_requests_compared": len(hashes_a),
        "scope": baseline["scope"],
        "note": (
            "Replay completion and hash comparison are not an Agent reward evaluation"
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
