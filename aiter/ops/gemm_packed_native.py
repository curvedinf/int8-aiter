# SPDX-License-Identifier: Apache-2.0
"""Lazy HIP kernel for packed-weight decode on gfx908."""

import ctypes
import fcntl
import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch


@functools.lru_cache(maxsize=1)
def _load():
    root = Path(__file__).resolve().parents[2]
    source = root / "csrc/kernels/gemm_packed_g128_gfx908.hip"
    compiler = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin/hipcc"
    version = subprocess.check_output([str(compiler), "--version"])
    digest = hashlib.sha256(
        source.read_bytes() + version + b"-O3-contract-off-v1"
    ).hexdigest()
    cache = Path.home() / ".cache/aiter/packed_g128_gfx908" / digest
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / "kernel.so"
    with (cache / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = cache / "kernel.tmp.so"
            with (cache / "build.log").open("w") as log:
                subprocess.run(
                    [
                        str(compiler),
                        "-O3",
                        "-ffp-contract=off",
                        "-fno-strict-aliasing",
                        "--offload-arch=gfx908",
                        "-shared",
                        "-fPIC",
                        str(source),
                        "-o",
                        str(temporary),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            temporary.replace(library)
    module = ctypes.CDLL(str(library))
    fn = module.launch_packed
    fn.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 9 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return module, fn


def packed_gemm(
    q,
    scales,
    packed,
    weight_scales,
    global_scale,
    groups,
    grid,
    out,
    m,
    n,
    k,
    start,
    length,
    split,
    codec,
):
    """Launch partial sums; the caller retains the GS128 output epilogue."""
    _, fn = _load()
    mt = 1 if m <= 16 or (codec != "nvfp4" and n <= 512) else 2
    status = fn(
        *[
            t.data_ptr()
            for t in (q, scales, packed, weight_scales, global_scale, groups, grid, out)
        ],
        m,
        n,
        k,
        start,
        length,
        int(codec == "nvfp4" and global_scale.numel() != 1),
        split,
        mt,
        {"nvfp4": 0, "q8_0": 1, "iq2_xxs": 2}[codec],
        torch.cuda.current_stream(q.device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"Packed GS128 HIP launch failed: {status}")


def packed_experts(
    q, scales, packed, groups, grid, out, plan, routes, n, k, start, length, topk, down
):
    module, _ = _load()
    fn = module.launch_routed
    fn.argtypes = (
        [ctypes.c_void_p] * 9 + [ctypes.c_int] * 8 + [ctypes.c_int64, ctypes.c_void_p]
    )
    fn.restype = ctypes.c_int
    sorted_ids, experts, count = plan
    status = fn(
        *[
            t.data_ptr()
            for t in (q, scales, packed, groups, grid, out, sorted_ids, experts, count)
        ],
        routes,
        n,
        k,
        start,
        length,
        experts.numel(),
        topk,
        int(down),
        packed.stride(0),
        torch.cuda.current_stream(q.device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"Packed expert HIP launch failed: {status}")
