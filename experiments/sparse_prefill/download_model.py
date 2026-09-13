# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download pinned official Qwen files with bounded, resumable range requests."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["Qwen3-0.6B", "Qwen3-4B"], required=True)
    parser.add_argument("--hf-manifest", type=Path, required=True)
    parser.add_argument("--modelscope-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    hf = json.loads(args.hf_manifest.read_text())
    metadata = json.loads(args.modelscope_manifest.read_text())
    assert hf["id"] == f"Qwen/{args.model}"
    hf_files = {entry["rfilename"]: entry for entry in hf["siblings"]}
    names = {
        "config.json",
        "generation_config.json",
        "LICENSE",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "model.safetensors.index.json",
    }
    files = [
        entry
        for entry in metadata["Data"]["Files"]
        if entry["Name"] in names or entry["Name"].endswith(".safetensors")
    ]
    if args.metadata_only:
        files = [entry for entry in files if not entry["Name"].endswith(".safetensors")]
    for entry in files:
        name = entry["Name"]
        assert Path(name).name == name and name in hf_files
        assert entry["Size"] == hf_files[name]["size"]
        if "lfs" in hf_files[name]:
            assert entry["Sha256"] == hf_files[name]["lfs"]["sha256"]

    args.output.mkdir(parents=True, exist_ok=True)
    chunk_size = 32 * 1024 * 1024
    for entry in files:
        name, size = entry["Name"], entry["Size"]
        target = args.output / name
        if target.exists() and sha256(target) == entry["Sha256"]:
            print("Verified cached", name, flush=True)
            continue
        query = urllib.parse.urlencode(
            {"Revision": entry["Revision"], "FilePath": name}
        )
        url = f"https://modelscope.cn/api/v1/models/Qwen/{args.model}/repo?{query}"
        part_dir = args.output / f".{name}.parts"
        part_dir.mkdir(exist_ok=True)

        def fetch(index, size=size, part_dir=part_dir, url=url, name=name):
            start = index * chunk_size
            end = min(start + chunk_size, size) - 1
            part = part_dir / str(index)
            if part.exists() and part.stat().st_size == end - start + 1:
                return part
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                expected_range = f"bytes {start}-{end}/{size}"
                if response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(f"Server did not honor range {expected_range}")
                with part.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
            if part.stat().st_size != end - start + 1:
                raise RuntimeError(f"Incomplete range: {name} part {index}")
            print("Downloaded range", name, start, end, flush=True)
            return part

        count = (size + chunk_size - 1) // chunk_size
        with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            parts = list(pool.map(fetch, range(count)))
        temporary = target.with_name(target.name + ".assembling")
        with temporary.open("wb") as output:
            for part in parts:
                with part.open("rb") as stream:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
        if sha256(temporary) != entry["Sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {name}; partial evidence retained")
        temporary.replace(target)
        for part in parts:
            part.unlink()
        part_dir.rmdir()
        print("Verified", name, size, flush=True)
    (args.output / "download-manifest.json").write_text(
        json.dumps(
            {
                "hf_revision": hf["sha"],
                "metadata_only": args.metadata_only,
                "files": files,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
