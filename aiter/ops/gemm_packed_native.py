# SPDX-License-Identifier: Apache-2.0
"""Lazy HIP kernel for packed NVFP4 decode on gfx908."""

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
    source = root / "csrc/kernels/gemm_packed_nvfp4_gfx908.hip"
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
    fn = module.launch_nv4
    fn.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 8 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return module, fn


def packed_nvfp4(
    q,
    scales,
    packed,
    weight_scales,
    global_scale,
    groups,
    out,
    m,
    n,
    k,
    start,
    length,
    split,
):
    """Launch partial sums; the caller retains the GS128 output epilogue."""
    _, fn = _load()
    mt = 1 if m <= 16 else 2
    status = fn(
        *[
            t.data_ptr()
            for t in (q, scales, packed, weight_scales, global_scale, groups, out)
        ],
        m,
        n,
        k,
        start,
        length,
        int(global_scale.numel() != 1),
        split,
        mt,
        torch.cuda.current_stream(q.device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"Packed NVFP4 HIP launch failed: {status}")
