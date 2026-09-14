# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve raw logs losslessly and fingerprint the final experiment artifacts."""

import gzip
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    for path in (directory / "service_results").rglob("*.log"):
        compressed = gzip.compress(path.read_bytes(), mtime=0)
        target = path.with_suffix(".log.gz")
        if target.exists():
            assert gzip.decompress(target.read_bytes()) == path.read_bytes()
        else:
            target.write_bytes(compressed)
    analysis = json.loads((directory / "service_analysis.json").read_text())
    code = analysis["rounds"][0]["summary"]["code_sha"]
    for name, expected in code.items():
        assert digest((root / name).read_bytes()) == expected, name
    files = {}
    for path in sorted(directory.rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix in (".log", ".pyc")
            or path.name in ("artifact_manifest.json", ".DS_Store")
        ):
            continue
        data = path.read_bytes()
        entry = dict(sha256=digest(data), bytes=len(data))
        if path.suffix == ".gz":
            entry["uncompressed_sha256"] = digest(gzip.decompress(data))
        files[str(path.relative_to(directory))] = entry
    manifest = dict(
        scope="Final code and evidence; raw logs are lossless deterministic gzip",
        tested_production_files=code,
        files=files,
    )
    (directory / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(dict(artifact_files=len(files), production_source_match=True)))


if __name__ == "__main__":
    main()
