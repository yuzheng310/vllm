# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the preregistered CPU-only service matrix inside the experiment root."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backend", choices=["hf", "fastokens"], required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    repo = root / "src/vllm-prompt-encoding"
    exp = root / "experiments/prompt-encoding"
    scripts = repo / "experiments/prompt_encoding"
    for label, budget in (("a1", 0), ("b1", 16), ("b2", 16), ("a2", 0)):
        stem = exp / "results" / f"{args.backend}_{label}"
        if stem.with_suffix(".log").exists():
            raise FileExistsError(stem)
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES="",
            TOKENIZERS_PARALLELISM="true",
            RAYON_NUM_THREADS="4",
            VLLM_CHATML_ENCODING_CACHE_MB=str(budget),
            VLLM_USE_FASTOKENS=str(int(args.backend == "fastokens")),
            REPOCOMPASS_RENDER_METRICS=str(stem.with_suffix(".renderer.json")),
            PYTHONPATH=str(repo) + os.pathsep + str(exp / "fastokens"),
        )
        command = [
            sys.executable,
            str(scripts / "serve_instrumented.py"),
            "launch",
            "render",
            str(root / "models/Qwen3-4B"),
            "--tokenizer",
            str(exp / "input/tokenizer"),
            "--served-model-name",
            "repocompass",
            "--host",
            "127.0.0.1",
            "--port",
            "18193",
            "--max-model-len",
            "40960",
            "--renderer-num-workers",
            "8",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--disable-uvicorn-access-log",
        ]
        with stem.with_suffix(".log").open("w") as log:
            server = subprocess.Popen(
                command, cwd=repo, env=env, stdout=log, stderr=log
            )
            try:
                with httpx.Client(timeout=1, trust_env=False) as client:
                    deadline = time.monotonic() + 120
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError(f"Server exited: {stem}.log")
                        try:
                            if (
                                client.get("http://127.0.0.1:18193/health").status_code
                                == 200
                            ):
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() > deadline:
                            raise TimeoutError("Render server startup")
                        time.sleep(0.25)
                subprocess.run(
                    [
                        sys.executable,
                        str(scripts / "bench_service.py"),
                        "--prepared",
                        str(exp / "input/repocompass-chat-workload.json"),
                        "--url",
                        "http://127.0.0.1:18193",
                        "--output",
                        str(stem.with_suffix(".http.json")),
                    ],
                    cwd=repo,
                    env=env,
                    check=True,
                )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            if server.returncode not in (0, -15):
                raise RuntimeError(f"Unexpected exit {server.returncode}")
        print(
            json.dumps(dict(backend=args.backend, round=label, status="complete")),
            flush=True,
        )


if __name__ == "__main__":
    main()
