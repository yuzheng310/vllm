# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Describe all preregistered E2E rounds without treating requests as replicates."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (position - low) * (values[high] - values[low])


def counters(text):
    result = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        series, value = line.rsplit(" ", 1)
        name = series.split("{", 1)[0]
        if name.endswith(("_sum", "_count", "_total")):
            result[name] = result.get(name, 0) + float(value)
    return result


def output_text(row):
    text = {}
    tools = {}
    for delta in row["deltas"]:
        for key in ("content", "reasoning", "reasoning_content"):
            if delta.get(key):
                text[key] = text.get(key, "") + delta[key]
        for tool in delta.get("tool_calls", []):
            index = tool["index"]
            target = tools.setdefault(index, {})
            for key, value in tool.get("function", {}).items():
                if value:
                    target[key] = target.get(key, "") + value
    # Tool call IDs are server metadata, not model output; ignore chunk boundaries.
    return {"text": text, "tools": tools}


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.input.read_text())
    assert report["status"] == "complete"
    assert [r["label"] for r in report["rounds"]] == ["a1", "b1", "b2", "a2"]
    summaries = {}
    outputs = {}
    for run in report["rounds"]:
        requests = [r for t in run["tasks"] for r in t["requests"]]
        assert len(run["tasks"]) == 48 and len(requests) == 233
        assert sum(r["usage"]["completion_tokens"] for r in requests) == 22046
        assert len(run["renderer"]["records"]) == 233
        if run["cache_mb"]:
            assert run["reset"]["cache"]["entries"] == 0
            assert run["renderer"]["cache"]["hits"] > 0
        else:
            assert run["renderer"]["cache"] is None
        first = [
            r["first_content_s"] for r in requests if r["first_content_s"] is not None
        ]
        before, after = counters(run["metrics_before"]), counters(run["metrics_after"])
        summaries[run["label"]] = dict(
            elapsed_s=run["elapsed_s"],
            trajectories_per_min=run["trajectories_per_min"],
            request_p50_s=statistics.median(r["elapsed_s"] for r in requests),
            request_p95_s=percentile([r["elapsed_s"] for r in requests], 0.95),
            first_content_p50_s=statistics.median(first) if first else None,
            first_content_p95_s=percentile(first, 0.95) if first else None,
            first_content_count=len(first),
            batch_completion_p95_s=percentile(
                [t["completed_s"] for t in run["tasks"]], 0.95
            ),
            tokenize_ms=sum(r["elapsed_ns"] for r in run["renderer"]["records"]) / 1e6,
            audit_ms=sum(r["audit_ns"] for r in run["renderer"]["records"]) / 1e6,
            api_process_cpu_s=(run["renderer"]["cpu_ns"] - run["reset"]["cpu_ns"])
            / 1e9,
            cache=run["renderer"]["cache"],
            native_counter_deltas={
                key: value - before.get(key, 0) for key, value in after.items()
            },
        )
        outputs[run["label"]] = {
            (r["task_id"], r["turn_id"]): output_text(r) for r in requests
        }
    comparisons = []
    for a, b in (("a1", "b1"), ("a2", "b2"), ("a1", "a2"), ("b1", "b2")):
        av, bv = summaries[a]["elapsed_s"], summaries[b]["elapsed_s"]
        comparisons.append(
            dict(
                a=a,
                b=b,
                elapsed_reduction_pct=100 * (1 - bv / av),
                throughput_increase_pct=100 * (av / bv - 1),
                elapsed_saved_s=av - bv,
                same_output_text_requests=sum(
                    outputs[a][key] == outputs[b][key] for key in outputs[a]
                ),
            )
        )
    a = statistics.median(summaries[k]["elapsed_s"] for k in ("a1", "a2"))
    b = statistics.median(summaries[k]["elapsed_s"] for k in ("b1", "b2"))
    result = dict(
        source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        scope=report["scope"],
        rounds=summaries,
        comparisons=comparisons,
        group_medians=dict(
            native_s=a,
            cache_s=b,
            saved_s=a - b,
            elapsed_reduction_pct=100 * (1 - b / a),
            throughput_increase_pct=100 * (a / b - 1),
        ),
        interpretation=(
            "Two repeats per arm; descriptive only, no independence or significance "
            "claim. Inspect resources and actual KV capacity before attribution."
        ),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["group_medians"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
