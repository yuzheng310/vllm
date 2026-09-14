# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze expected IDs after native API tool-schema serialization; keep bodies."""

import argparse
import hashlib
import json
from pathlib import Path

from bench_cpu import digest
from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--first-renderer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Canonical input/report already exists")
    frozen = json.loads(args.prepared.read_text())
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    observations = []
    for item in frozen["requests"]:
        request = ChatCompletionRequest(
            model="repocompass",
            messages=item["messages"],
            tools=item["tools"],
            max_tokens=1,
            add_special_tokens=False,
        )
        tools = (
            None
            if request.tools is None
            else [tool.model_dump() for tool in request.tools]
        )
        # dict equality disregards key order; ensure no tools were removed or changed.
        assert tools == item["tools"]
        text = tok.apply_chat_template(
            item["messages"],
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        ids = tok(text, add_special_tokens=False)["input_ids"]
        observations.append(
            dict(
                task_id=item["task_id"],
                turn_id=item["turn_id"],
                original_text_sha256=hashlib.sha256(item["text"].encode()).hexdigest(),
                canonical_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                original_tokens=len(item["input_ids"]),
                canonical_tokens=len(ids),
                ids_equal=ids == item["input_ids"],
                tool_values_unchanged=True,
            )
        )
        item["text"] = text
        item["input_ids"] = ids
    if args.first_renderer is not None:
        observed = json.loads(args.first_renderer.read_text())["requests"]
        assert observations[0]["canonical_text_sha256"] in {
            row["text_sha256"] for row in observed
        }, "Tool serialization alone does not explain the native discrepancy"
    frozen["canonicalization"] = dict(
        original_prepared_sha256=digest(args.prepared),
        method="Native API tool model_dump; request bodies unchanged",
    )
    args.output.write_text(json.dumps(frozen, ensure_ascii=False) + "\n")
    report = dict(
        method=frozen["canonicalization"],
        original_native_observation=(
            None if args.first_renderer is None else digest(args.first_renderer)
        ),
        canonical_prepared_sha256=digest(args.output),
        changed_id_sequences=sum(not row["ids_equal"] for row in observations),
        requests=len(observations),
        observations=observations,
    )
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "observations"}))


if __name__ == "__main__":
    main()
