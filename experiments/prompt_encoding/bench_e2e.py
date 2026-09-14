# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay complete historical chats and recorded decode lengths over GPU HTTP."""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from array import array
from collections import Counter
from pathlib import Path

import httpx


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_digest(ids):
    packed = array("I", ids)
    assert packed.itemsize == 4
    if sys.byteorder != "little":
        packed.byteswap()
    return hashlib.sha256(packed.tobytes()).hexdigest()


def prepare(prepared, workload):
    frozen = json.loads(prepared.read_text())
    source = json.loads(workload.read_text())
    assert digest(workload) == frozen["workload_sha256"]
    index = {(item["task_id"], item["turn_id"]): item for item in frozen["requests"]}
    tasks = []
    for task in source["tasks"]:
        requests = []
        for row in task["requests"]:
            item = index[(task["task_id"], row["turn_id"])]
            length = row["recorded_output_tokens"]
            assert len(item["input_ids"]) + length <= 40960
            requests.append(
                dict(
                    task_id=task["task_id"],
                    turn_id=row["turn_id"],
                    input_tokens=len(item["input_ids"]),
                    output_tokens=length,
                    ids_sha256=ids_digest(item["input_ids"]),
                    text_sha256=hashlib.sha256(item["text"].encode()).hexdigest(),
                    body=dict(
                        model="repocompass",
                        messages=item["messages"],
                        tools=item["tools"],
                        max_tokens=length,
                        add_special_tokens=False,
                        temperature=0,
                        seed=42,
                        ignore_eos=True,
                        stream=True,
                        stream_options={"include_usage": True},
                    ),
                )
            )
        tasks.append(dict(task_id=task["task_id"], requests=requests))
    assert len(tasks) == 48
    assert sum(len(task["requests"]) for task in tasks) == 233
    assert sum(r["output_tokens"] for t in tasks for r in t["requests"]) == 22046
    return tasks


async def request(client, item, warmup=False):
    body = dict(item["body"])
    if warmup:
        body["max_tokens"] = 16
    # Measure serialization, HTTP, GPU and all response chunks together.
    begun = time.perf_counter()
    first_content = None
    deltas = []
    usage = None
    done = False
    async with client.stream("POST", "/v1/chat/completions", json=body) as response:
        if response.status_code != 200:
            raise RuntimeError((await response.aread()).decode()[:2000])
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                done = True
                continue
            chunk = json.loads(payload)
            if "error" in chunk:
                raise RuntimeError(chunk)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content = {
                    key: value
                    for key, value in delta.items()
                    if key != "role" and value not in (None, "", [])
                }
                if content:
                    if first_content is None:
                        first_content = time.perf_counter()
                    deltas.append(content)
    elapsed = time.perf_counter() - begun
    assert done and usage is not None, "Incomplete SSE/usage"
    assert usage["prompt_tokens"] == item["input_tokens"], usage
    assert usage["completion_tokens"] == body["max_tokens"], usage
    # Preserve content for inspecting generation differences; not a quality score.
    return dict(
        task_id=item["task_id"],
        turn_id=item["turn_id"],
        elapsed_s=elapsed,
        first_content_s=(None if first_content is None else first_content - begun),
        usage=usage,
        deltas=deltas,
    )


async def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    tasks = prepare(args.prepared, args.workload)
    expected = Counter(
        (row["text_sha256"], row["ids_sha256"], row["input_tokens"])
        for task in tasks
        for row in task["requests"]
    )
    report = dict(
        status="running",
        scope="Historical multi-turn GPU HTTP replay; simulated tool delay; no RL",
        prepared_sha256=digest(args.prepared),
        workload_sha256=digest(args.workload),
        script_sha256=digest(Path(__file__)),
        concurrency=4,
        tool_delay_s=0.25,
        rounds=[],
    )

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    async with httpx.AsyncClient(
        base_url=args.url,
        timeout=httpx.Timeout(180),
        trust_env=False,
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
    ) as client:
        try:
            longest = max(
                (r for t in tasks for r in t["requests"]),
                key=lambda r: r["input_tokens"],
            )
            await request(client, longest, warmup=True)
            await asyncio.gather(
                *(request(client, t["requests"][0], warmup=True) for t in tasks[:4])
            )
            report["warmup_requests"] = 5
            save()
            for label, budget in (("a1", 0), ("b1", 16), ("b2", 16), ("a2", 0)):
                reset = await client.post(
                    "/__experiment/reset", params={"cache_mb": budget, "label": label}
                )
                reset.raise_for_status()
                before = reset.json()
                if budget:
                    assert before["cache"]["entries"] == 0
                metrics_before = (await client.get("/metrics")).text
                pending = asyncio.Queue()
                for task in tasks:
                    pending.put_nowait(task)
                rows = []
                begun = time.perf_counter()

                async def worker(pending=pending, rows=rows, begun=begun):
                    while not pending.empty():
                        task = pending.get_nowait()
                        task_start = time.perf_counter()
                        requests = []
                        for turn, item in enumerate(task["requests"]):
                            if turn:
                                await asyncio.sleep(0.25)
                            start = time.perf_counter()
                            result = await request(client, item)
                            result["start_s"] = start - begun
                            requests.append(result)
                        rows.append(
                            dict(
                                task_id=task["task_id"],
                                elapsed_s=time.perf_counter() - task_start,
                                completed_s=time.perf_counter() - begun,
                                requests=requests,
                            )
                        )
                        pending.task_done()

                await asyncio.wait_for(
                    asyncio.gather(*(worker() for _ in range(4))), timeout=600
                )
                elapsed = time.perf_counter() - begun
                snapshot = await client.get("/__experiment/snapshot")
                snapshot.raise_for_status()
                renderer = snapshot.json()
                actual = Counter(
                    (r["text_sha256"], r["ids_sha256"], r["tokens"])
                    for r in renderer["records"]
                )
                assert actual == expected, "Actual renderer token sequence mismatch"
                assert len(rows) == 48
                metrics_after = (await client.get("/metrics")).text
                result = dict(
                    label=label,
                    cache_mb=budget,
                    elapsed_s=elapsed,
                    trajectories_per_min=48 * 60 / elapsed,
                    start_ns=int(begun * 1e9),
                    reset=before,
                    renderer=renderer,
                    metrics_before=metrics_before,
                    metrics_after=metrics_after,
                    tasks=rows,
                )
                report["rounds"].append(result)
                save()
                print(
                    json.dumps(
                        {
                            k: result[k]
                            for k in ("label", "elapsed_s", "trajectories_per_min")
                        }
                    ),
                    flush=True,
                )
            report["status"] = "complete"
        except BaseException as exc:
            report["status"] = "failed"
            report["error"] = repr(exc)
            raise
        finally:
            save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
