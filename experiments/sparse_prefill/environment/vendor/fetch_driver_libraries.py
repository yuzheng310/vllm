# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gzip
import hashlib
import json
import pathlib
import urllib.request

root = pathlib.Path(__file__).resolve().parent
base = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/"
version = "580.173.02-1ubuntu1"
wanted = {"libnvidia-compute-580", "libnvidia-gpucomp-580", "nvidia-utils-580"}
print("Reading official package checksums", flush=True)
with urllib.request.urlopen(base + "Packages.gz", timeout=30) as response:
    index = gzip.decompress(response.read()).decode()
selected = []
for block in index.split("\n\n"):
    fields = dict(
        line.split(": ", 1)
        for line in block.splitlines()
        if ": " in line and not line.startswith(" ")
    )
    if (
        fields.get("Package") in wanted
        and fields.get("Version") == version
        and fields.get("Architecture") == "amd64"
    ):
        selected.append(
            {
                key: fields[key]
                for key in ("Package", "Version", "Filename", "Size", "SHA256")
            }
        )
assert {item["Package"] for item in selected} == wanted, selected
(root / "driver-packages.json").write_text(json.dumps(selected, indent=2))
for item in selected:
    filename = pathlib.PurePosixPath(item["Filename"]).name
    target = root / filename
    if (
        target.exists()
        and hashlib.sha256(target.read_bytes()).hexdigest() == item["SHA256"]
    ):
        print("Verified existing", filename, flush=True)
        continue
    print("Downloading", filename, "bytes", item["Size"], flush=True)
    digest = hashlib.sha256()
    received = 0
    with (
        urllib.request.urlopen(base + item["Filename"], timeout=30) as response,
        target.open("wb") as output,
    ):
        while True:
            data = response.read(4 * 1024 * 1024)
            if not data:
                break
            output.write(data)
            digest.update(data)
            received += len(data)
            print("Progress", received, "/", item["Size"], flush=True)
    assert received == int(item["Size"])
    assert digest.hexdigest() == item["SHA256"], filename
    print("SHA256 verified", filename, flush=True)
print("DOWNLOADS_VERIFIED", flush=True)
