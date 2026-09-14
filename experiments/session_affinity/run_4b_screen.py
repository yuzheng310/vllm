# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the predeclared three-arm 4B screen inside the isolated lab workspace."""

import argparse
import json
import math
import os
import statistics
import subprocess
from pathlib import Path

BASELINE_SHA = "98dff2a81d747d1dba01a47f939f48c3526d4206"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-name", default="session-4b-screen-20260914")
    args = parser.parse_args()
    root = args.root.resolve()
    if Path(args.output_name).name != args.output_name:
        raise ValueError("Output name must be one directory component")
    output = root / "reports" / args.output_name
    output.mkdir(exist_ok=False)
    candidate = root / "src/vllm-session-affinity"
    baseline = root / "src/vllm-session-baseline"
    script = candidate / "experiments/session_affinity/replay.py"
    model = root / "models/Qwen3-4B"
    manifest = json.loads((model / "verified-session-weights.json").read_text())
    expected_weights = {e["Name"]: e["Sha256"] for e in manifest["files"]}
    workload = root / "experiments/session-affinity-workload.json"
    tasks = json.loads(workload.read_text())["tasks"]
    previous = {
        (task["task_id"], turn): len(task["requests"][turn - 1]["input_ids"])
        for task in tasks
        for turn in range(1, len(task["requests"]))
    }
    for tree in (baseline, candidate):
        dirty = subprocess.check_output(
            ["git", "-C", str(tree), "status", "--porcelain"], text=True
        ).strip()
        if dirty:
            raise RuntimeError(f"Tracked/untracked changes in {tree}")
    actual_baseline = subprocess.check_output(
        ["git", "-C", str(baseline), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_baseline != BASELINE_SHA:
        raise RuntimeError("Unexpected release baseline")
    record = {"status": "running", "commands": [], "results": {}}

    def save():
        (output / "screen.json").write_text(json.dumps(record, indent=2) + "\n")

    def run(name, policy, concurrency, max_wait=0.5):
        tree = candidate if policy == "affinity" else baseline
        destination = output / f"{name}.json"
        command = [
            str(root / ".venv/bin/python"),
            str(script),
            "--model",
            str(model),
            "--workload",
            str(workload),
            "--output",
            str(destination),
            "--policy",
            policy,
            "--concurrency",
            str(concurrency),
            "--max-wait",
            str(max_wait),
            "--runs",
            "1",
        ]
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(tree),
            OMP_NUM_THREADS="1",
            HF_HUB_OFFLINE="1",
            VLLM_BATCH_INVARIANT="0",
        )
        record["commands"].append(
            {
                "name": name,
                "argv": [arg.replace(str(root) + "/", "") for arg in command],
                "pythonpath": str(tree.relative_to(root)),
                "omp_num_threads": 1,
                "batch_invariant": 0,
            }
        )
        save()
        print(f"Starting {name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
            )
        report = json.loads(destination.read_text())
        if report["status"] != "complete":
            raise RuntimeError("Incomplete arm")
        if report["weights_sha256"] != expected_weights:
            raise RuntimeError("Weights differ from the verified manifest")
        result = report["runs"][0]
        shortfall = sum(
            max(
                0,
                previous[(task["task_id"], request["turn"])] // 16 * 16
                - request["cached_tokens"],
            )
            for task in result["tasks"]
            for request in task["requests"]
            if request["turn"]
        )
        summary = {
            "wall_s": result["elapsed_s"],
            "completed_tasks": result["completed_tasks"],
            "previous_input_token_shortfall": shortfall,
            "cache_hits": result["cache_hits"],
            "cache_queries": result["cache_queries"],
            "preemptions": result["preemptions"],
        }
        record["results"][name] = summary
        save()
        print(json.dumps({name: summary}), flush=True)
        return result, summary

    try:
        native8, summary = run("fcfs-c8-a1", "fcfs", 8)
        median_queue = statistics.median(
            r["queued_time"] for r in native8["finished_request_metrics"]
        )
        max_wait = math.ceil(median_queue)
        record.update(queue_median_s=median_queue, affinity_max_wait_s=max_wait)
        save()
        if summary["previous_input_token_shortfall"] == 0:
            record["status"] = "stopped_no_previous_input_recompute"
            return
        _, native4 = run("fcfs-c4-a1", "fcfs", 4)
        _, affinity = run("affinity-c8-b1", "affinity", 8, max_wait)
        best_native = min(summary["wall_s"], native4["wall_s"])
        reduction = 100 * (1 - affinity["wall_s"] / best_native)
        record.update(
            status="screen_complete",
            wall_reduction_vs_faster_native_percent=reduction,
            repeat_candidate_if_tail_and_completion_pass=reduction >= 10,
        )
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        save()
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
