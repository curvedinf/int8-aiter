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


@functools.lru_cache(maxsize=1)
def _hadamard_module():
    module = ctypes.CDLL(str(_build("head_hadamard_bf16_gfx908.hip")))
    fn = module.launch_head_hadamard
    fn.argtypes = [ctypes.c_void_p] * 5
    fn.restype = ctypes.c_int
    return fn


def head_hadamard_supported(x: torch.Tensor) -> bool:
    return (
        _arch() == "gfx908"
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.shape[-1] % 64 == 0
        and x.is_contiguous()
    )


def head_hadamard(x: torch.Tensor, out: torch.Tensor | None = None):
    """Normalized 64-point Hadamard over the last dim; self-inverse.

    Returns a new tensor when ``out`` is None, else writes into ``out``
    (which may alias ``x``). Falls back to None when unsupported.
    """
    if not head_hadamard_supported(x):
        return None
    dst = torch.empty_like(x) if out is None else out
    status = _hadamard_module()(
        x.data_ptr(),
        dst.data_ptr(),
        x.numel() // x.shape[-1],
        x.shape[-1],
        torch.cuda.current_stream().cuda_stream,
    )
    if status != 0:
        raise RuntimeError(f"head hadamard launch failed: hip status {status}")
    return dst


@functools.lru_cache(maxsize=1)
def _attn_module():
    module = ctypes.CDLL(str(_build("paged_attn_decode_int8_pthead_gfx908.hip")))
    pack = module.launch_pack_q
    pack.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64, ctypes.c_void_p]
    pack.restype = ctypes.c_int
    attn = module.launch_paged_attn_decode_int8
    attn.argtypes = (
        [ctypes.c_void_p] * 12
        + [ctypes.c_int] * 7
        + [ctypes.c_int64] * 12
        + [ctypes.c_float, ctypes.c_void_p]
    )
    attn.restype = ctypes.c_int
    red = module.launch_reduce_segments
    red.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int64] + [ctypes.c_int] * 3
        + [ctypes.c_void_p]
    )
    red.restype = ctypes.c_int
    return pack, attn, red


def paged_attn_decode_int8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    softmax_scale: float,
    num_segments: int = 8,
) -> None:
    """Paged int8 decode attention for gfx908 (int8 per-token-head cache).

    ``q`` is bf16 [tokens, q_heads, dim] (already Hadamard-rotated by the
    caller when rotation is on); K/V are int8 with per-(token, head) fp32
    scales. Writes bf16 into ``out`` (same shape as ``q``).
    """
    assert q.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    assert max_seqlen_q <= 8, "native kernel covers verify blocks up to 8"
    num_tokens, num_q_heads, dim = q.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    qg = num_q_heads // num_kv_heads
    rows = num_tokens * num_q_heads
    dev = q.device

    q8 = torch.empty((rows, dim), dtype=torch.int8, device=dev)
    qsc = torch.empty((rows,), dtype=torch.float32, device=dev)
    pack, attn, red = _attn_module()
    stream = torch.cuda.current_stream().cuda_stream
    status = pack(q.data_ptr(), q8.data_ptr(), qsc.data_ptr(), rows, stream)
    if status != 0:
        raise RuntimeError(f"pack_q failed: {status}")

    seg_out = torch.empty((rows, num_segments, dim), dtype=torch.float32, device=dev)
    seg_m = torch.empty((rows, num_segments), dtype=torch.float32, device=dev)
    seg_l = torch.empty_like(seg_m)

    kb, ks, kv = k_cache.stride(0), k_cache.stride(1), k_cache.stride(2)
    vb, vs, vv = v_cache.stride(0), v_cache.stride(1), v_cache.stride(2)
    ksc_b, ksc_s, ksc_h = k_scale_cache.stride()
    vsc_b, vsc_s, vsc_h = v_scale_cache.stride()
    status = attn(
        q8.data_ptr(), qsc.data_ptr(),
        k_cache.data_ptr(), v_cache.data_ptr(),
        k_scale_cache.data_ptr(), v_scale_cache.data_ptr(),
        block_table.data_ptr(), seqused_k.data_ptr(), cu_seqlens_q.data_ptr(),
        seg_out.data_ptr(), seg_m.data_ptr(), seg_l.data_ptr(),
        cu_seqlens_q.numel() - 1, num_segments, num_kv_heads, num_q_heads, qg,
        block_size, block_table.shape[1],
        kb, ks, kv, vb, vs, vv,
        ksc_b, ksc_s, ksc_h, vsc_b, vsc_s, vsc_h,
        float(softmax_scale), stream,
    )
    if status != 0:
        raise RuntimeError(f"paged_attn_decode_int8 failed: {status}")
    status = red(
        seg_out.data_ptr(), seg_m.data_ptr(), seg_l.data_ptr(), out.data_ptr(),
        rows, cu_seqlens_q.numel() - 1, num_q_heads, num_segments, stream,
    )
    if status != 0:
        raise RuntimeError(f"reduce_segments failed: {status}")
