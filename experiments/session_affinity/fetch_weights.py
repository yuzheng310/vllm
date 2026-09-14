# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resume verified Qwen weights using small ranges on a timeout-prone link."""

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_file(entry, output, legacy_dirs):
    name, size = entry["Name"], entry["Size"]
    target = output / name
    if target.exists() and sha256(target) == entry["Sha256"]:
        print("Verified cached", name, flush=True)
        return
    chunk = 4 * 1024 * 1024
    parts = output / f".{name}.chunks-4m"
    parts.mkdir(exist_ok=True)
    for root in [output, *legacy_dirs]:
        old_parts = root / f".{name}.parts"
        if not old_parts.exists():
            continue
        for old in old_parts.iterdir():
            if not old.name.isdecimal():
                continue
            offset = int(old.name) * 32 * 1024 * 1024
            with old.open("rb") as stream:
                while data := stream.read(chunk):
                    expected = min(chunk, size - offset)
                    if expected <= 0 or len(data) != expected:
                        break
                    part = parts / str(offset // chunk)
                    if not part.exists() or part.stat().st_size != expected:
                        part.write_bytes(data)
                    offset += len(data)
    url = f"https://modelscope.cn/models/Qwen/Qwen3-4B/resolve/master/{name}"
    count = (size + chunk - 1) // chunk

    def fetch(index):
        start, end = index * chunk, min((index + 1) * chunk, size) - 1
        part = parts / str(index)
        if part.exists() and part.stat().st_size == end - start + 1:
            return part
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, headers={"Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(req, timeout=30) as response:
                    expected_range = f"bytes {start}-{end}/{size}"
                    if response.headers.get("Content-Range") != expected_range:
                        raise ValueError("Unexpected range response")
                    with part.open("wb") as stream:
                        shutil.copyfileobj(response, stream, length=256 * 1024)
                if part.stat().st_size != end - start + 1:
                    raise ValueError("Incomplete range")
                return part
            except (OSError, ValueError) as error:
                print("Retry", name, index, attempt, repr(error), flush=True)
                if attempt == 2:
                    raise
                time.sleep(1)
        raise AssertionError("Unreachable")

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, _ in enumerate(pool.map(fetch, range(count)), 1):
            if done % 32 == 0 or done == count:
                print(
                    name,
                    done,
                    "/",
                    count,
                    "elapsed",
                    time.monotonic() - started,
                    flush=True,
                )
    temporary = target.with_name(name + ".assembling-4m")
    with temporary.open("wb") as stream:
        for index in range(count):
            with (parts / str(index)).open("rb") as source:
                shutil.copyfileobj(source, stream, length=1024 * 1024)
    if sha256(temporary) != entry["Sha256"]:
        raise ValueError(f"Whole-file SHA256 mismatch: {name}")
    temporary.replace(target)
    print("Verified", name, size, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-manifest", type=Path, required=True)
    parser.add_argument("--modelscope-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    hf = json.loads(args.hf_manifest.read_text())
    assert hf["id"] == "Qwen/Qwen3-4B"
    hf_files = {entry["rfilename"]: entry for entry in hf["siblings"]}
    metadata = json.loads(args.modelscope_manifest.read_text())
    files = [e for e in metadata["Data"]["Files"] if e["Name"].endswith(".safetensors")]
    assert len(files) == 3
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in files:
        name = entry["Name"]
        assert Path(name).name == name
        assert entry["Sha256"] == hf_files[name]["lfs"]["sha256"]
        assert entry["Size"] == hf_files[name]["size"]
        fetch_file(entry, args.output, args.legacy_dir)
    (args.output / "verified-session-weights.json").write_text(
        json.dumps({"hf_revision": hf["sha"], "files": files}, indent=2)
    )


if __name__ == "__main__":
    main()
