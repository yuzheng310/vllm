# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the real GPU-less vLLM server, measuring its unchanged renderer call."""

import atexit
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

from vllm.entrypoints.launchers.render import entry as render_entry
from vllm.renderers.base import BaseRenderer

records = []
renderers = []
original = BaseRenderer._tokenize_prompt
original_build_app = render_entry.build_app


def measured(self, prompt, params):
    cpu_start = time.process_time_ns()
    start = time.perf_counter_ns()
    result = original(self, prompt, params)
    elapsed = time.perf_counter_ns() - start
    cpu_elapsed = time.process_time_ns() - cpu_start
    records.append(
        dict(
            start_ns=start,
            elapsed_ns=elapsed,
            cpu_ns=cpu_elapsed,
            text_sha256=hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
            tokens=len(result["prompt_token_ids"]),
            offsets=params.return_token_offsets,
        )
    )
    if not renderers:
        renderers.append(self)
    return result


def save():
    output = Path(os.environ["REPOCOMPASS_RENDER_METRICS"])
    if output.exists():
        return
    renderer = renderers[0] if renderers else None
    cache = None if renderer is None else renderer._chatml_encoding_cache
    output.write_text(
        json.dumps(
            dict(
                scope="Real vLLM launch render; no model inference",
                commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                diff_sha256=hashlib.sha256(
                    subprocess.check_output(["git", "diff", "HEAD"])
                ).hexdigest(),
                code_sha256={
                    name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                    for name in (
                        "vllm/envs.py",
                        "vllm/renderers/base.py",
                        "vllm/tokenizers/chatml_encoding_cache.py",
                    )
                },
                platform=platform.platform(),
                python=platform.python_version(),
                versions={
                    name: importlib.metadata.version(name)
                    for name in ("transformers", "tokenizers", "vllm")
                },
                backend=(
                    None
                    if renderer is None
                    else str(type(renderer.get_tokenizer().backend_tokenizer))
                ),
                environment={
                    k: os.environ.get(k)
                    for k in (
                        "VLLM_CHATML_ENCODING_CACHE_MB",
                        "VLLM_USE_FASTOKENS",
                        "TOKENIZERS_PARALLELISM",
                        "RAYON_NUM_THREADS",
                        "CUDA_VISIBLE_DEVICES",
                    )
                },
                cache=None if cache is None else cache.snapshot(),
                requests=records,
            ),
            indent=2,
        )
        + "\n"
    )


BaseRenderer._tokenize_prompt = measured
atexit.register(save)


def instrumented_app(*args, **kwargs):
    app = original_build_app(*args, **kwargs)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with original_lifespan(app):
            try:
                yield
            finally:
                save()

    app.router.lifespan_context = lifespan
    return app


render_entry.build_app = instrumented_app

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
