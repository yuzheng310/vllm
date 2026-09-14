# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run fixed-prompt, fixed-length multi-turn replay on an actual async engine."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import vllm
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.loggers import StatLoggerBase


class ReplayStats(StatLoggerBase):
    latest = None

    def __init__(self, vllm_config, engine_index=0):
        type(self).latest = self
        self.reset()

    def reset(self):
        self.queries = self.hits = self.preemptions = 0
        self.max_waiting = 0
        self.finished = []

    def log_engine_initialized(self):
        pass

    def record(
        self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0
    ):
        if scheduler_stats is not None:
            cache = scheduler_stats.prefix_cache_stats
            self.queries += cache.queries
            self.hits += cache.hits
            self.max_waiting = max(self.max_waiting, scheduler_stats.num_waiting_reqs)
        if iteration_stats is not None:
            self.preemptions += iteration_stats.num_preempted_reqs
            self.finished.extend(
                asdict(item) for item in iteration_stats.finished_requests
            )


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def main(args):
    if args.output.exists():
        raise ValueError("Refusing to overwrite results")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if gpu:
        raise RuntimeError("GPU already has a compute process; refusing to contend")
    workload = json.loads(args.workload.read_text())
    index = args.model / "model.safetensors.index.json"
    weight_names = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else ["model.safetensors"]
    )
    weights = {name: digest(args.model / name) for name in weight_names}
    tasks = workload["tasks"][: args.limit_tasks or None]
    engine_args = AsyncEngineArgs(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=workload["max_model_len"],
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=True,
        disable_cascade_attn=True,
        skip_tokenizer_init=True,
        enforce_eager=args.eager,
        disable_log_stats=False,
        scheduler_cls=(
            "vllm.v1.core.sched.session_affinity_scheduler.SessionAffinityScheduler"
            if args.policy == "affinity"
            else None
        ),
        session_affinity_max_wait_s=args.max_wait,
        seed=42,
    )
    started = time.perf_counter()
    llm = AsyncLLM.from_engine_args(engine_args, stat_loggers=[ReplayStats])
    report = {
        "status": "running",
        "policy": args.policy,
        "model": args.model.name,
        "weights_sha256": weights,
        "workload_sha256": digest(args.workload),
        "script_sha256": digest(Path(__file__)),
        "concurrency": args.concurrency,
        "tool_delay_s": args.tool_delay,
        "affinity_max_wait_s": args.max_wait,
        "engine_args": {
            "max_num_seqs": 8,
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.8,
            "max_model_len": workload["max_model_len"],
            "async": True,
            "apc": True,
            "eager": args.eager,
        },
        "engine_startup_s": time.perf_counter() - started,
        "scope": "historical prompt replay; fixed decode lengths; simulated tool delay",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(Path(vllm.__file__).parents[1]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, default=str))

    try:
        warm_params = SamplingParams(
            temperature=0, max_tokens=8, ignore_eos=True, detokenize=False
        )
        async for _ in llm.generate(
            {"prompt_token_ids": tasks[0]["requests"][0]["input_ids"]},
            warm_params,
            "warmup",
            session_id="warmup",
        ):
            pass
        for run in range(args.runs):
            if not await llm.reset_prefix_cache():
                raise RuntimeError("Cold cache reset failed")
            stats = ReplayStats.latest
            assert stats is not None
            stats.reset()
            pending = asyncio.Queue()
            for task in tasks:
                pending.put_nowait(task)
            rows = []
            begun = time.perf_counter()

            async def worker(pending=pending, rows=rows, begun=begun, run=run):
                while not pending.empty():
                    task = pending.get_nowait()
                    task_begin = time.perf_counter()
                    requests = []
                    for turn, request in enumerate(task["requests"]):
                        if turn:
                            await asyncio.sleep(args.tool_delay)
                        begin = time.perf_counter()
                        first = None
                        output = None
                        async for output in llm.generate(
                            {"prompt_token_ids": request["input_ids"]},
                            SamplingParams(
                                temperature=0,
                                max_tokens=request["recorded_output_tokens"],
                                ignore_eos=True,
                                detokenize=False,
                                seed=42,
                            ),
                            f"{run}/{task['task_id']}/{turn}",
                            session_id=f"{run}/{task['task_id']}",
                        ):
                            if first is None and output.outputs[0].token_ids:
                                first = time.perf_counter()
                        assert (
                            output is not None and output.finished and first is not None
                        )
                        ids = output.outputs[0].token_ids
                        if len(ids) != request["recorded_output_tokens"]:
                            raise RuntimeError("Unexpected decode length")
                        requests.append(
                            {
                                "turn": turn,
                                "start_s": begin - begun,
                                "elapsed_s": time.perf_counter() - begin,
                                "ttft_s": first - begin,
                                "input_tokens": len(request["input_ids"]),
                                "output_tokens": len(ids),
                                "cached_tokens": output.num_cached_tokens,
                                "output_sha256": hashlib.sha256(
                                    np.asarray(ids, dtype="<i4").tobytes()
                                ).hexdigest(),
                            }
                        )
                    rows.append(
                        {
                            "task_id": task["task_id"],
                            "elapsed_s": time.perf_counter() - task_begin,
                            "completed_s": time.perf_counter() - begun,
                            "requests": requests,
                        }
                    )
                    pending.task_done()

            await asyncio.gather(*(worker() for _ in range(args.concurrency)))
            elapsed = time.perf_counter() - begun
            result = {
                "run": run,
                "elapsed_s": elapsed,
                "trajectories_per_min": len(rows) * 60 / elapsed,
                "trajectory_p95_s": float(
                    np.percentile([row["elapsed_s"] for row in rows], 95)
                ),
                "completed_tasks": len(rows),
                "cache_queries": stats.queries,
                "cache_hits": stats.hits,
                "max_waiting": stats.max_waiting,
                "preemptions": stats.preemptions,
                "finished_request_metrics": stats.finished,
                "tasks": rows,
            }
            report["runs"].append(result)
            save()
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in result.items()
                        if k not in ("tasks", "finished_request_metrics")
                    }
                ),
                flush=True,
            )
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        save()
        llm.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=["fcfs", "affinity"], required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--tool-delay", type=float, default=0.25)
    parser.add_argument("--max-wait", type=float, default=0.5)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--eager", action="store_true")
    asyncio.run(main(parser.parse_args()))
