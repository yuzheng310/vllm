# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the fixed single-GPU matrix inside the original isolated workspace."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import httpx
import psutil


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gpu_pids():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    return {int(line.strip()) for line in output.splitlines() if line.strip()}


def resources(server=None):
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,"
            "temperature.gpu,clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    values = raw.split(", ")
    if len(raw.splitlines()) != 1:
        raise RuntimeError("Expected exactly one GPU")
    processes = []
    if server is not None and server.poll() is None:
        parent = psutil.Process(server.pid)
        for proc in [parent, *parent.children(recursive=True)]:
            with suppress(psutil.NoSuchProcess):
                processes.append(dict(pid=proc.pid, rss=proc.memory_info().rss))
    return dict(
        time_ns=time.perf_counter_ns(),
        gpu_csv=raw,
        gpu_free_mib=float(values[2]),
        mem_available_bytes=psutil.virtual_memory().available,
        gpu_pids=sorted(gpu_pids()),
        server_processes=processes,
    )


def stop(process):
    if process is None:
        return
    # Include children after an unexpectedly exited parent. Only our newly
    # created session has this process-group ID; psutil also guards PID reuse.
    group = []
    for proc in psutil.process_iter():
        with suppress(ProcessLookupError, PermissionError, psutil.NoSuchProcess):
            if os.getpgid(proc.pid) == process.pid:
                group.append(proc)
    for proc in group:
        with suppress(psutil.NoSuchProcess):
            proc.terminate()
    _, alive = psutil.wait_procs(group, timeout=15)
    for proc in alive:
        with suppress(psutil.NoSuchProcess):
            proc.kill()
    process.wait(timeout=15)


def main(args):
    root = args.root.resolve(strict=True)
    if root.name != "vllm-sparse-prefill.JbyyBUjL":
        raise ValueError("This protocol is restricted to the authorized workspace")
    for variable in ("TMPDIR", "HF_HOME", "VLLM_CACHE_ROOT", "PRE_COMMIT_HOME"):
        path = Path(os.environ.get(variable, "/")).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"Use run-in-workspace.sh: {variable} is not isolated")
    if not Path(sys.prefix).resolve().is_relative_to(root):
        raise RuntimeError("Use the workspace Python environment")
    repo = root / "src/vllm-prompt-encoding"
    scripts = repo / "experiments/prompt_encoding"
    inputs = root / "experiments/prompt-encoding/input"
    output = args.output.resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Output must be a new directory within the workspace")
    output.mkdir(parents=True)
    metadata = dict(
        status="preflight", protocol_sha256=digest(scripts / "E2E_PROTOCOL.md")
    )

    def save():
        (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    server = client_proc = None
    server_log = client_log = samples = None
    save()
    try:
        initial = resources()
        metadata["preflight"] = initial
        save()
        if initial["gpu_pids"]:
            raise RuntimeError("GPU already occupied; no model will be started")
        if initial["gpu_free_mib"] < 21 * 1024:
            raise RuntimeError("Less than 21 GiB GPU memory available")
        if initial["mem_available_bytes"] < 12 * 1024**3:
            raise RuntimeError("Less than 12 GiB system memory available")
        with socket.socket() as port_check:
            port_check.bind(("127.0.0.1", 18194))
        model = root / "models/Qwen3-4B"
        manifest = json.loads((model / "verified-session-weights.json").read_text())
        actual_weights = {
            e["Name"]: digest(model / e["Name"]) for e in manifest["files"]
        }
        if actual_weights != {e["Name"]: e["Sha256"] for e in manifest["files"]}:
            raise RuntimeError("Weights fail previously fixed SHA256 manifest")
        prepared = inputs / "repocompass-chat-workload-native.json"
        workload = root / "experiments/session-affinity-workload.json"
        from bench_e2e import prepare

        tasks = prepare(prepared, workload)
        metadata.update(
            weights_sha256=actual_weights,
            tokenizer_sha256={
                p.name: digest(p)
                for p in (inputs / "tokenizer").glob("*")
                if p.is_file()
            },
            model_config_sha256=digest(model / "config.json"),
            prepared_sha256=digest(prepared),
            workload_sha256=digest(workload),
            base_commit=subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip(),
            diff_sha256=hashlib.sha256(
                subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD"])
            ).hexdigest(),
            code_sha256={
                name: digest(repo / name)
                for name in (
                    "vllm/envs.py",
                    "vllm/renderers/base.py",
                    "vllm/tokenizers/chatml_encoding_cache.py",
                    "experiments/prompt_encoding/serve_e2e.py",
                    "experiments/prompt_encoding/bench_e2e.py",
                    "experiments/prompt_encoding/run_e2e.py",
                )
            },
            versions={
                name: importlib.metadata.version(name)
                for name in ("vllm", "torch", "transformers", "tokenizers")
            },
            task_count=len(tasks),
        )
        del tasks
        if metadata["base_commit"] != "98dff2a81d747d1dba01a47f939f48c3526d4206":
            raise RuntimeError("Expected the isolated v0.29.0 source base")
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES="0",
            HF_HUB_OFFLINE="1",
            OMP_NUM_THREADS="1",
            TOKENIZERS_PARALLELISM="true",
            RAYON_NUM_THREADS="4",
            VLLM_USE_FASTOKENS="1",
            VLLM_CHATML_ENCODING_CACHE_MB="0",
            VLLM_BATCH_INVARIANT="0",
            PYTHONPATH=str(repo) + os.pathsep + str(inputs.parent / "fastokens"),
        )
        command = [
            sys.executable,
            str(scripts / "serve_e2e.py"),
            "serve",
            str(model),
            "--tokenizer",
            str(inputs / "tokenizer"),
            "--served-model-name",
            "repocompass",
            "--host",
            "127.0.0.1",
            "--port",
            "18194",
            "--api-server-count",
            "1",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "40960",
            "--max-num-seqs",
            "8",
            "--max-num-batched-tokens",
            "2048",
            "--kv-cache-memory-bytes",
            str(10 * 1024**3),
            "--gpu-memory-utilization",
            "0.8",
            "--cpu-offload-gb",
            "0",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--async-scheduling",
            "--renderer-num-workers",
            "8",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--seed",
            "42",
            "--generation-config",
            "vllm",
            "--disable-uvicorn-access-log",
        ]
        metadata.update(
            command=command,
            environment={
                k: env[k]
                for k in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "RAYON_NUM_THREADS",
                    "VLLM_USE_FASTOKENS",
                    "VLLM_CHATML_ENCODING_CACHE_MB",
                    "PYTHONPATH",
                )
            },
        )
        save()
        server_log = (output / "server.log").open("x")
        client_log = (output / "client.log").open("x")
        samples = (output / "resources.jsonl").open("x")
        server = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=server_log,
            stderr=server_log,
            start_new_session=True,
        )
        metadata.update(status="starting", server_pid=server.pid)
        save()
        begun = time.monotonic()
        last_sample = 0.0

        def monitor():
            nonlocal last_sample
            if server.poll() is not None:
                raise RuntimeError(f"Server exited {server.returncode}")
            if time.monotonic() - last_sample < 5:
                return
            row = resources(server)
            samples.write(json.dumps(row) + "\n")
            samples.flush()
            last_sample = time.monotonic()
            own = {r["pid"] for r in row["server_processes"]}
            if set(row["gpu_pids"]) - own:
                raise RuntimeError("External GPU process appeared; stop our experiment")
            if row["mem_available_bytes"] < 4 * 1024**3:
                raise RuntimeError("System available memory below 4 GiB")

        with httpx.Client(
            base_url="http://127.0.0.1:18194", timeout=1, trust_env=False
        ) as http:
            # Reject port collisions before relying on health alone.
            while True:
                monitor()
                try:
                    response = http.get("/__experiment/snapshot")
                    if (
                        response.status_code == 200
                        and response.json()["pid"] == server.pid
                    ):
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() - begun > 600:
                    raise TimeoutError("GPU server initialization exceeds 10 minutes")
                time.sleep(0.5)
        metadata.update(status="running", startup_s=time.monotonic() - begun)
        save()
        client_proc = subprocess.Popen(
            [
                sys.executable,
                str(scripts / "bench_e2e.py"),
                "--prepared",
                str(prepared),
                "--workload",
                str(workload),
                "--url",
                "http://127.0.0.1:18194",
                "--output",
                str(output / "replay.json"),
            ],
            cwd=repo,
            env=env,
            stdout=client_log,
            stderr=client_log,
            start_new_session=True,
        )
        deadline = time.monotonic() + 3000
        while client_proc.poll() is None:
            monitor()
            if time.monotonic() > deadline:
                raise TimeoutError("Replay including warmup exceeds 50 minutes")
            time.sleep(0.5)
        if client_proc.returncode != 0:
            raise RuntimeError(
                f"Replay exited {client_proc.returncode}; retain client log"
            )
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        stop(client_proc)
        stop(server)
        with suppress(Exception):
            metadata["after_shutdown"] = resources()
        metadata["server_returncode"] = None if server is None else server.returncode
        metadata["client_returncode"] = (
            None if client_proc is None else client_proc.returncode
        )
        for stream in (samples, client_log, server_log):
            if stream is not None:
                stream.close()
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
