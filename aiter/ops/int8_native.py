# SPDX-License-Identifier: Apache-2.0
"""Lane-native int8 kernels for gfx908: loader, self-test, and dispatch gates.

The kernels live in ``csrc/kernels/*.hip`` and are compiled on first use with
the same cached-hipcc pattern as ``aiter.ops.gemm_packed_native``. Shared
fragment primitives are in ``csrc/kernels/int8_mfma_lane.cuh``.

Design: ``docs/design/int8_small_gemm_native.md`` in the vLLM fork.
Scope: int8 only.
"""

import ctypes
import fcntl
import functools
import hashlib
import os
import subprocess
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
_KERNELS = _ROOT / "csrc/kernels"

# Sites are enabled one at a time as each kernel clears its roofline gate.
SITES = ("ua_decode", "gemm_m16", "lm_head")


def _arch() -> str:
    return os.environ.get("INT8_NATIVE_ARCH", "gfx908")


def _build(source_name: str, version_tag: str = "v1") -> Path:
    """Compile ``csrc/kernels/<source_name>`` once per source/compiler digest."""
    source = _KERNELS / source_name
    compiler = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "bin/hipcc"
    version = subprocess.check_output([str(compiler), "--version"])
    digest = hashlib.sha256(
        source.read_bytes()
        + (_KERNELS / "int8_mfma_lane.cuh").read_bytes()
        + version
        + f"-O3-{version_tag}-{_arch()}".encode()
    ).hexdigest()
    cache = Path.home() / ".cache/aiter/int8_native" / source.stem / digest
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
                        f"--offload-arch={_arch()}",
                        "-I",
                        str(_KERNELS),
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
    return library


@functools.lru_cache(maxsize=None)
def _selftest_module():
    module = ctypes.CDLL(str(_build("int8_mfma_selftest.hip")))
    fn = module.launch_int8_selftest
    fn.argtypes = [ctypes.c_void_p] * 6
    fn.restype = ctypes.c_int
    return fn


def selftest() -> dict:
    """Run the bit-exact primitive checks; raise on any mismatch."""
    torch.manual_seed(0)
    a = torch.randint(-128, 128, (16, 16), dtype=torch.int8, device="cuda")
    b = torch.randint(-128, 128, (16, 16), dtype=torch.int8, device="cuda")
    c_direct = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    c_lds = torch.zeros((16, 16), dtype=torch.int32, device="cuda")
    colsum = torch.zeros((16,), dtype=torch.float32, device="cuda")

    fn = _selftest_module()
    status = fn(
        a.data_ptr(),
        b.data_ptr(),
        c_direct.data_ptr(),
        c_lds.data_ptr(),
        colsum.data_ptr(),
        torch.cuda.current_stream().cuda_stream,
    )
    if status != 0:
        raise RuntimeError(f"int8 selftest launch failed: hip status {status}")

    ref = (a.to(torch.float32) @ b.to(torch.float32)).to(torch.int32)
    checks = {
        "lane_maps": bool(torch.equal(c_direct, ref)),
        "lds_row16": bool(torch.equal(c_lds, ref)),
        "token_reduce": bool(
            torch.equal(colsum, ref.sum(dim=0).to(torch.float32))
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"int8 MFMA selftest failed: {checks}")
    return checks


def native_sites() -> set:
    """Sites enabled by ``AITER_INT8_NATIVE`` (default: none)."""
    raw = os.environ.get("AITER_INT8_NATIVE", "0").strip().lower()
    if raw in ("", "0", "false", "off"):
        return set()
    if raw in ("1", "true", "on", "all"):
        return set(SITES)
    return {s.strip() for s in raw.split(",") if s.strip() in SITES}


def native_enabled(site: str) -> bool:
    return site in native_sites()
