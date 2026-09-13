# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from pathlib import Path

import torch

import vllm
from vllm.platforms import current_platform

assert torch.accelerator.is_available()
x = torch.arange(1024, device="cuda", dtype=torch.float32)
assert torch.equal((x * 3 + 7).cpu(), torch.arange(1024, dtype=torch.float32) * 3 + 7)
report = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "vllm": vllm.__version__,
    "vllm_file": vllm.__file__,
    "gpu": current_platform.get_device_name(),
    "torch_compute": "PASS",
}
Path("reports/package-cuda-probe.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
