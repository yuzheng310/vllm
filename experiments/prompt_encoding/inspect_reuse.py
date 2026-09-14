# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit real ChatML segments and exact IDs; collect no performance measurements."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    workload = json.loads(args.workload.read_text())
    reconstruction = args.source / "reconstruct_tokens.py"
    if digest(reconstruction) != workload["reconstruction_sha256"]:
        raise ValueError("Reconstruction script differs from frozen workload")
    spec = importlib.util.spec_from_file_location("reconstruction", reconstruction)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parquet = args.source / "raw/swe_bench_verified.parquet"
    if digest(parquet) != workload["source_parquet_sha256"]:
        raise ValueError("Source parquet changed")
    wanted = {task["task_id"] for task in workload["tasks"]}
    rows = {
        row["instance_id"]: row
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=32)
        for row in batch.to_pylist()
        if row["instance_id"] in wanted
    }
    tokenizer_path = args.source / "tokenizer"
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    delimiter = "<|im_end|>"
    raw_tokenizer = json.loads((tokenizer_path / "tokenizer.json").read_text())
    marker = next(t for t in raw_tokenizer["added_tokens"] if t["content"] == delimiter)
    assert marker["special"] and not any(
        marker[k] for k in ("normalized", "single_word", "lstrip", "rstrip")
    )
    assert tok.encode(delimiter, add_special_tokens=False) == [marker["id"]]
    observations = []
    for task in workload["tasks"]:
        chat = rows[task["task_id"]]["chat_messages"]
        messages = [
            module._normalize_message(m) for m in module._as_list(chat["messages"])
        ]
        turns = module.conversation_turns(messages)
        assert len(turns) == len(task["requests"])
        seen = {}
        for request, (prefix, _) in zip(task["requests"], turns, strict=True):
            text = tok.apply_chat_template(
                prefix,
                tools=module._as_list(chat.get("tools")) or None,
                add_generation_prompt=True,
                tokenize=False,
            )
            full = tok(text, add_special_tokens=False)["input_ids"]
            if full != request["input_ids"]:
                raise ValueError(
                    f"Frozen input differs: {task['task_id']}/{request['turn_id']}"
                )
            parts = text.split(delimiter)
            joined, reused_bytes, reused_tokens, segment_bytes = [], 0, 0, 0
            for index, part in enumerate(parts):
                ids = tok(part, add_special_tokens=False)["input_ids"]
                size = len(part.encode("utf-8"))
                segment_bytes += size
                if part in seen:
                    reused_bytes += size
                    reused_tokens += len(ids)
                    assert seen[part] == ids
                seen[part] = ids
                joined.extend(ids)
                if index + 1 < len(parts):
                    joined.append(marker["id"])
            if joined != full:
                raise ValueError(
                    f"Segment encoding differs: {task['task_id']}/{request['turn_id']}"
                )
            observations.append(
                {
                    "task_id": task["task_id"],
                    "turn_id": request["turn_id"],
                    "prompt_tokens": len(full),
                    "prompt_utf8_bytes": len(text.encode("utf-8")),
                    "segment_count": len(parts),
                    "segment_utf8_bytes": segment_bytes,
                    "reused_segment_utf8_bytes": reused_bytes,
                    "reused_segment_tokens": reused_tokens,
                    "ideal_cache_payload_bytes": sum(
                        len(s.encode("utf-8")) + 4 * len(ids) for s, ids in seen.items()
                    ),
                    "rendered_text_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                }
            )
    import tokenizers
    import transformers

    summary = {
        "tasks": len(wanted),
        "requests": len(observations),
        "exact_frozen_inputs": len(observations),
        "exact_segment_encodings": len(observations),
        "prompt_tokens": sum(q["prompt_tokens"] for q in observations),
        "reused_segment_tokens": sum(q["reused_segment_tokens"] for q in observations),
        "segment_utf8_bytes": sum(q["segment_utf8_bytes"] for q in observations),
        "reused_segment_utf8_bytes": sum(
            q["reused_segment_utf8_bytes"] for q in observations
        ),
        "max_ideal_cache_payload_bytes_per_task": max(
            q["ideal_cache_payload_bytes"] for q in observations
        ),
    }
    result = {
        "scope": (
            "Structural reuse only; per-trajectory cache, no eviction, cold first "
            "turns included. No measured speedup or production memory claim."
        ),
        "workload_sha256": digest(args.workload),
        "source_parquet_sha256": digest(parquet),
        "reconstruction_sha256": digest(reconstruction),
        "analyzer_sha256": digest(Path(__file__)),
        "tokenizer_revision": workload["tokenizer_revision"],
        "tokenizer_json_sha256": digest(tokenizer_path / "tokenizer.json"),
        "chat_template_sha256": digest(tokenizer_path / "chat_template.jinja"),
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "delimiter_definition": marker,
        "normalizer": raw_tokenizer["normalizer"],
        "summary": summary,
        "requests": observations,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
