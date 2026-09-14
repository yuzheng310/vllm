# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize complete 4B artifacts without pooling different numeric modes."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p95(values):
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    workload_hash = digest(args.workload)
    workload = json.loads(args.workload.read_text())
    capacity_path = args.directory / "capacity-audit.json"
    capacities = {
        entry["run"]: entry for entry in json.loads(capacity_path.read_text())
    }
    previous = {}
    for task in workload["tasks"]:
        for turn, request in enumerate(task["requests"]):
            if turn:
                prev = task["requests"][turn - 1]["input_ids"]
                if request["input_ids"][: len(prev)] != prev:
                    raise ValueError("Input is not append-only")
                previous[(task["task_id"], turn)] = len(prev)
    rows, output_hashes = [], {}
    for path in sorted(args.directory.glob("*.json")):
        report = json.loads(path.read_text())
        if "runs" not in report:
            continue
        if report["status"] != "complete":
            raise ValueError(f"Incomplete report: {path}")
        if report["workload_sha256"] != workload_hash:
            raise ValueError(f"Different workload: {path}")
        capacity = capacities[path.stem]
        for run in report["runs"]:
            requests = [q for t in run["tasks"] for q in t["requests"]]
            shortfall = affected = 0
            hashes = {}
            for task in run["tasks"]:
                for request in task["requests"]:
                    key = (task["task_id"], request["turn"])
                    hashes[key] = request["output_sha256"]
                    if request["turn"]:
                        gap = max(
                            0, previous[key] // 16 * 16 - request["cached_tokens"]
                        )
                        shortfall += gap
                        affected += int(gap > 0)
            name = f"{path.stem}/run{run['run']}"
            output_hashes[name] = hashes
            rows.append(
                {
                    "name": name,
                    "report_sha256": digest(path),
                    "policy": report["policy"],
                    "concurrency": report["concurrency"],
                    "max_wait_s": report["affinity_max_wait_s"],
                    "model": report["model"],
                    "weights_sha256": report["weights_sha256"],
                    "script_sha256": report["script_sha256"],
                    "source_commit": report["source_commit"],
                    "engine_args": report["engine_args"],
                    "kv_pool_tokens": capacity["kv_pool_tokens"],
                    "allocation_log_sha256": capacity["log_sha256"],
                    "batch_invariant": report["batch_invariant"],
                    "omp_num_threads": report["omp_num_threads"],
                    "wall_s": run["elapsed_s"],
                    "trajectories_per_min": run["trajectories_per_min"],
                    "batch_completion_p95_s": p95(
                        [t["completed_s"] for t in run["tasks"]]
                    ),
                    "trajectory_active_p95_s": run["trajectory_p95_s"],
                    "first_request_p95_s": p95(
                        [q["elapsed_s"] for q in requests if q["turn"] == 0]
                    ),
                    "ttft_p95_s": p95([q["ttft_s"] for q in requests]),
                    "cache_hits": run["cache_hits"],
                    "cache_queries": run["cache_queries"],
                    "previous_input_token_shortfall": shortfall,
                    "later_requests_with_shortfall": affected,
                    "completed_tasks": run["completed_tasks"],
                    "requests": len(requests),
                    "preemptions": run["preemptions"],
                    "corrupted_requests": sum(
                        bool(m["is_corrupted"]) for m in run["finished_request_metrics"]
                    ),
                }
            )
    comparisons = []
    for index, a in enumerate(rows):
        for b in rows[index + 1 :]:
            comparable = all(
                a[key] == b[key]
                for key in (
                    "model",
                    "weights_sha256",
                    "script_sha256",
                    "engine_args",
                    "kv_pool_tokens",
                    "batch_invariant",
                    "omp_num_threads",
                )
            )
            if not comparable:
                continue
            x, y = output_hashes[a["name"]], output_hashes[b["name"]]
            if x.keys() != y.keys():
                raise ValueError("Different request identities")
            comparisons.append(
                {
                    "a": a["name"],
                    "b": b["name"],
                    "output_sequence_matches": sum(x[k] == y[k] for k in x),
                    "requests_compared": len(x),
                    "note": "Concurrency may differ; not a task quality evaluation",
                }
            )
    args.output.write_text(
        json.dumps(
            {
                "workload_sha256": workload_hash,
                "analyzer_sha256": digest(Path(__file__)),
                "capacity_audit_sha256": digest(capacity_path),
                "rows": rows,
                "comparable_output_checks": comparisons,
                "scope": "Historical prompt replay, simulated tool delay, no live RL",
                "note": (
                    "Do not pool cascade settings, batch-invariant modes, or actual "
                    "KV capacities. Equal gpu_memory_utilization is insufficient. "
                    "Request phase durations are not GPU operator time."
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
