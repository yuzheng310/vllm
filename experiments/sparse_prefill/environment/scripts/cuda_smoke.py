# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes as c
import json
from pathlib import Path

cuda = c.CDLL("libcuda.so.1")


def bind(name, types):
    fn = getattr(cuda, name)
    fn.argtypes = types
    fn.restype = c.c_int
    return fn


bind("cuGetErrorName", [c.c_int, c.POINTER(c.c_char_p)])


def check(code):
    if code:
        msg = c.c_char_p()
        cuda.cuGetErrorName(code, c.byref(msg))
        raise RuntimeError((code, msg.value))


check(bind("cuInit", [c.c_uint])(0))
dev = c.c_int()
check(bind("cuDeviceGet", [c.POINTER(c.c_int), c.c_int])(c.byref(dev), 0))
ctx, module, function = c.c_void_p(), c.c_void_p(), c.c_void_p()
device_mem = c.c_uint64()
check(
    bind("cuCtxCreate_v2", [c.POINTER(c.c_void_p), c.c_uint, c.c_int])(
        c.byref(ctx), 0, dev
    )
)
try:
    count = 1024
    check(
        bind("cuMemAlloc_v2", [c.POINTER(c.c_uint64), c.c_size_t])(
            c.byref(device_mem), count * 4
        )
    )
    check(
        bind("cuMemsetD32_v2", [c.c_uint64, c.c_uint, c.c_size_t])(device_mem, 0, count)
    )
    ptx = b"""\n.version 7.0
.target sm_80
.address_size 64
.visible .entry smoke(.param .u64 output) {
.reg .b32 %r<5>;
.reg .b64 %rd<4>;
ld.param.u64 %rd1, [output];
mov.u32 %r1, %tid.x;
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mad.lo.u32 %r1, %r2, %r3, %r1;
mul.wide.u32 %rd2, %r1, 4;
add.u64 %rd3, %rd1, %rd2;
mad.lo.u32 %r4, %r1, 3, 7;
st.global.u32 [%rd3], %r4;
ret;
}\n"""
    check(
        bind("cuModuleLoadData", [c.POINTER(c.c_void_p), c.c_char_p])(
            c.byref(module), ptx
        )
    )
    check(
        bind("cuModuleGetFunction", [c.POINTER(c.c_void_p), c.c_void_p, c.c_char_p])(
            c.byref(function), module, b"smoke"
        )
    )
    params = (c.c_void_p * 1)(c.addressof(device_mem))
    launch = bind(
        "cuLaunchKernel",
        [c.c_void_p]
        + [c.c_uint] * 7
        + [c.c_void_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)],
    )
    check(launch(function, 8, 1, 1, 128, 1, 1, 0, None, params, None))
    check(bind("cuCtxSynchronize", [])())
    result = (c.c_uint32 * count)()
    check(
        bind("cuMemcpyDtoH_v2", [c.c_void_p, c.c_uint64, c.c_size_t])(
            result, device_mem, count * 4
        )
    )
    assert list(result) == [i * 3 + 7 for i in range(count)]
    report = {
        "status": "PASS",
        "elements_verified": count,
        "device_buffer_bytes": count * 4,
        "checks": [
            "CUDA context",
            "device allocation and memset",
            "PTX JIT",
            "kernel launch",
            "synchronize",
            "device-to-host copy",
            "exact integer results",
        ],
    }
    Path("reports/cuda-smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
finally:
    if device_mem.value:
        check(bind("cuMemFree_v2", [c.c_uint64])(device_mem))
    if module.value:
        check(bind("cuModuleUnload", [c.c_void_p])(module))
    check(bind("cuCtxDestroy_v2", [c.c_void_p])(ctx))
