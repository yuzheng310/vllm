# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GPU server with experiment-only controls; bind to loopback only."""

import hashlib
import os
import sys
import time
from array import array

from fastapi import HTTPException

from vllm.entrypoints.launchers.api_server import entry
from vllm.renderers.base import BaseRenderer
from vllm.tokenizers.chatml_encoding_cache import ChatMLEncodingCache

records = []
phase = "warmup"
original_tokenize = BaseRenderer._tokenize_prompt
original_build = entry.build_app


def token_digest(ids):
    packed = array("I", ids)
    assert packed.itemsize == 4
    if sys.byteorder != "little":
        packed.byteswap()
    return hashlib.sha256(packed.tobytes()).hexdigest()


def measured(self, prompt, params):
    start = time.perf_counter_ns()
    result = original_tokenize(self, prompt, params)
    elapsed = time.perf_counter_ns() - start
    audit_start = time.perf_counter_ns()
    row = dict(
        phase=phase,
        start_ns=start,
        elapsed_ns=elapsed,
        text_sha256=hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        ids_sha256=token_digest(result["prompt_token_ids"]),
        tokens=len(result["prompt_token_ids"]),
    )
    row["audit_ns"] = time.perf_counter_ns() - audit_start
    records.append(row)
    return result


def build_app(*args, **kwargs):
    app = original_build(*args, **kwargs)

    @app.post("/__experiment/reset")
    async def reset(cache_mb: int, label: str):
        global phase
        if cache_mb not in (0, 16):
            raise HTTPException(400, "Only preregistered cache budgets are accepted")
        engine = app.state.engine_client
        if not await engine.reset_prefix_cache():
            raise HTTPException(409, "GPU cache still holds active requests")
        renderer = engine.renderer
        cache = (
            ChatMLEncodingCache.create(
                renderer.get_tokenizer(), max_bytes=cache_mb * 1024 * 1024
            )
            if cache_mb
            else None
        )
        if cache_mb and cache is None:
            raise HTTPException(400, "Tokenizer does not support this cache")
        renderer._chatml_encoding_cache = cache
        records.clear()
        phase = label
        return dict(
            label=phase,
            cache=None if cache is None else cache.snapshot(),
            cpu_ns=time.process_time_ns(),
        )

    @app.get("/__experiment/snapshot")
    async def snapshot():
        cache = app.state.engine_client.renderer._chatml_encoding_cache
        return dict(
            pid=os.getpid(),
            phase=phase,
            cpu_ns=time.process_time_ns(),
            cache=None if cache is None else cache.snapshot(),
            records=records,
        )

    return app


BaseRenderer._tokenize_prompt = measured
entry.build_app = build_app

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
