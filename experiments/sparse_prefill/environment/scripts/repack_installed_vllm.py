# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import importlib.metadata
import json
import zipfile
from pathlib import Path

import pybase64 as base64

root = Path.cwd()
distribution = importlib.metadata.distribution("vllm")
assert distribution.version == "0.29.0", distribution.version
output = root / "vendor/vllm-0.29.0-installed-artifacts.zip"
manifest = []
with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED) as archive:
    for item in distribution.files:
        name = str(item)
        if not name.startswith("vllm/") or "__pycache__" in name:
            continue
        source = Path(distribution.locate_file(item))
        assert source.is_file()
        assert source.resolve().is_relative_to(root / ".venv")
        digest = hashlib.sha256(source.read_bytes()).digest()
        if item.hash is not None:
            assert item.hash.mode == "sha256"
            assert (
                base64.urlsafe_b64encode(digest).decode().rstrip("=") == item.hash.value
            ), name
        archive.write(source, name)
        manifest.append(
            {"path": name, "bytes": source.stat().st_size, "sha256": digest.hex()}
        )
(root / "reports/precompiled-artifact-manifest.json").write_text(
    json.dumps(
        {
            "distribution": "vllm==0.29.0",
            "note": (
                "Local ZIP repacked from verified installed RECORD entries; "
                "not the original wheel"
            ),
            "files": manifest,
        },
        indent=2,
    )
)
print(
    "PRECOMPILED_ARTIFACTS_VERIFIED", len(manifest), output.stat().st_size, flush=True
)
