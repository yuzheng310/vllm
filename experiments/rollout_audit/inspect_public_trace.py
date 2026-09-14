# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inspect existing public rollout metadata without running inference.

Run with uv run --no-project --with pyarrow python <script> --help.
No prompt text is exported. Output creation is exclusive to preserve evidence.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    return {
        "n": len(ordered),
        "sum": sum(ordered),
        "p50": ordered[round((len(ordered) - 1) * 0.5)],
        "p95": ordered[round((len(ordered) - 1) * 0.95)],
        "max": ordered[-1],
    }


def inspect(path: Path) -> dict:
    import pyarrow.parquet as pq

    task_ids = []
    prompt_lengths = []
    output_lengths = []
    latencies = []
    cache_values: Counter = Counter()
    metric_keys: set[str] = set()
    message_keys: set[str] = set()
    latency_keys: set[str] = set()
    models: set[str] = set()
    requests_without_matching_latency = 0
    turns = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            task_ids.append(row["instance_id"])
            metrics = row.get("metrics") or {}
            metric_keys.update(metrics)
            usages = metrics.get("token_usages") or []
            turns.append(len(usages))
            records = metrics.get("response_latencies") or []
            latency_ids = {r.get("response_id") for r in records}
            for record in records:
                latency_keys.update(record)
                if record.get("latency") is not None:
                    latencies.append(record["latency"])
            for usage in usages:
                prompt_lengths.append(usage["prompt_tokens"])
                output_lengths.append(usage["completion_tokens"])
                cache = usage.get("cache_read_tokens", "missing")
                cache_values[str(cache)] += 1
                if usage.get("model"):
                    models.add(usage["model"])
                if usage.get("response_id") not in latency_ids:
                    requests_without_matching_latency += 1
            for message in (row.get("chat_messages") or {}).get("messages") or []:
                message_keys.update(message)
    return {
        "source_filename": path.name,
        "source_sha256": fingerprint(path),
        "n_tasks": len(task_ids),
        "n_unique_task_ids": len(set(task_ids)),
        "turns_per_task": distribution(turns),
        "input_tokens": distribution(prompt_lengths),
        "output_tokens": distribution(output_lengths),
        "recorded_request_latency_seconds": distribution(latencies),
        "input_tokens_above_40960": sum(n > 40960 for n in prompt_lengths),
        "input_tokens_above_16384": sum(n > 16384 for n in prompt_lengths),
        "cache_read_values": dict(cache_values),
        "requests_without_matching_latency_id": requests_without_matching_latency,
        "metric_fields": sorted(metric_keys),
        "latency_fields": sorted(latency_keys),
        "message_fields": sorted(message_keys),
        "model_labels": sorted(models),
        "limitations": [
            "Public evaluation trajectories, not current RepoCompass training logs.",
            "Cache telemetry zero does not establish engine APC misses.",
            "No engine queue/prefill/decode timing, tool timing or arrival trace.",
            "Summed request latency is not concurrent job completion time.",
            "This metadata inspection runs no inference or performance benchmark.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = inspect(args.parquet)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
