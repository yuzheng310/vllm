# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only guard: incomplete or shorter SSE must never become a speed result."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from analyze_e2e import output_text
from bench_e2e import prepare, request


async def check(item):
    async def run(mode):
        def respond(req):
            sent = json.loads(req.content)
            assert sent["max_tokens"] == item["output_tokens"]
            assert sent["messages"] == item["body"]["messages"]
            chunks = [
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"content": "中文"}}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": item["input_tokens"],
                        "completion_tokens": item["output_tokens"] - (mode == "short"),
                    },
                },
            ]
            content = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            if mode != "incomplete":
                content += "data: [DONE]\n\n"
            return httpx.Response(200, text=content)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="http://test.invalid"
        ) as client:
            return await request(client, item)

    valid = await run("valid")
    assert output_text(valid)["text"]["content"] == "中文"
    for mode in ("short", "incomplete"):
        try:
            await run(mode)
        except AssertionError:
            continue
        raise AssertionError(f"Invalid {mode} response was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    args = parser.parse_args()
    tasks = prepare(args.prepared, args.workload)
    asyncio.run(check(tasks[0]["requests"][0]))
    assert output_text({"deltas": [{"content": "a"}, {"content": "b"}]}) == output_text(
        {"deltas": [{"content": "ab"}]}
    )
    print(
        json.dumps(
            dict(
                status="PASS",
                tasks=48,
                requests=233,
                decode_tokens=22046,
                valid_sse=True,
                incomplete_rejected=True,
                short_decode_rejected=True,
                chunk_boundaries_ignored=True,
                scope="CPU contract checks only; no GPU or performance claim",
            )
        )
    )


if __name__ == "__main__":
    main()
