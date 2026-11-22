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
    cache_root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    cache = cache_root / "aiter" / "packed_g128_gfx908" / digest
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
    q,
    scales,
    packed,
    groups,
    grid,
    out,
    plan,
    routes,
    n,
    k,
    start,
    length,
    topk,
    down,
    cache=None,
):
    module, _ = _load()
    fn = module.launch_routed if cache is None else module.launch_routed_cached
    fn.argtypes = (
        [ctypes.c_void_p] * 9
        + [ctypes.c_int] * 9
        + [ctypes.c_int64]
        + ([ctypes.c_void_p] * 2 if cache is not None else [])
        + [ctypes.c_void_p]
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
        int(os.environ.get("VLLM_BOLT_ROUTED_WAVES", "2")),
        packed.stride(0),
        *(
            [cache.weight.data_ptr(), cache.mapping.data_ptr()]
            if cache is not None
            else []
        ),
        torch.cuda.current_stream(q.device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"Packed expert HIP launch failed: {status}")


def pack_affine_weights(weight, scale, zero=None, *, vector=False, overlap=False):
    """Pack N16/K16 microtiles for coalesced MFMA loads, keeping weight precision."""
    n = weight.shape[-2]
    k = scale.shape[-1] * 128
    bits = 4 if zero is not None else 8
    if n % 16 or weight.shape[-1] != k * bits // 8:
        raise ValueError("Invalid affine N16/K128 storage")
    leading = weight.shape[:-2]
    if overlap:
        if vector or n % 64:
            raise ValueError("The 32x64 MFMA layout requires whole 64-channel tiles")
        if bits == 4:
            # The low and high nibble hold signed codes for k and k+4.
            # This keeps scan order for both the MFMA and the vector dot.
            pairs = weight.reshape(*leading, n, k // 8, 4)
            lo, hi = pairs[..., :2], pairs[..., 2:]
            first = (lo & 15) | ((hi & 15) << 4)
            second = (lo >> 4) | (hi & 240)
            weight = torch.stack((first, second), dim=-1).flatten(-2) ^ 136
        tiles = weight.reshape(-1, n // 64, 64, k // 128, 8, 2 * bits)
        weight = tiles.permute(0, 1, 3, 4, 2, 5).contiguous().reshape(*leading, n, -1)

        def by_group(t):
            return t.transpose(-2, -1).contiguous().reshape_as(t)

        return weight, by_group(scale), None if zero is None else by_group(zero)
    # Expose 4 weights (2 or 4 bytes) per lane and keep all 64 lanes adjacent.
    word_bytes = bits // 2
    if vector:
        values = weight.reshape(-1, n // 16, 16, k // 128, 8, 4, word_bytes)
        values = values.permute(0, 1, 3, 5, 2, 4, 6)
    else:
        values = weight.reshape(-1, n // 16, 16, k // 16, 4, word_bytes)
        values = values.permute(0, 1, 3, 4, 2, 5)
    weight = values.contiguous().reshape(*leading, n, -1)

    def group_tiles(x):
        return (
            x.reshape(-1, n // 16, 16, k // 128)
            .transpose(-1, -2)
            .contiguous()
            .reshape(x.shape)
        )

    return weight, group_tiles(scale), None if zero is None else group_tiles(zero)


def dequantize_affine_weights(
    weight, scale, zero=None, *, wave_format=False, dtype=torch.float16
):
    """Recover [N,K] for the absorbed MLA projections, using the stored G128 grid."""
    n = weight.shape[-2]
    k = scale.shape[-1] * 128
    if wave_format == 3:
        group_count = k // 128
        bits = 4 if zero is not None else 8
        weight = weight.reshape(n // 64, group_count, 8, 64, 2 * bits)
        weight = weight.permute(0, 3, 1, 2, 4).reshape(n, -1)
        scale = scale.reshape(group_count, n).T
        if zero is not None:
            zero = zero.reshape(group_count, n).T
            pairs = (weight ^ 136).reshape(n, k // 8, 4)
            codes = torch.cat((pairs & 15, pairs >> 4), dim=-1).reshape(n, -1, 128)
            return (
                ((codes.float() - zero.unsqueeze(-1)) * scale.unsqueeze(-1))
                .reshape(n, k)
                .to(dtype)
            )
    elif wave_format:
        word_bytes = 2 if zero is not None else 4
        if wave_format == 2:
            weight = (
                weight.reshape(n // 16, k // 128, 4, 16, 8, word_bytes)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(n, -1)
            )
        else:
            weight = (
                weight.reshape(n // 16, k // 16, 4, 16, word_bytes)
                .permute(0, 3, 1, 2, 4)
                .reshape(n, -1)
            )

        def rows(x):
            return x.reshape(n // 16, k // 128, 16).transpose(1, 2).reshape(n, -1)

        scale = rows(scale)
        if zero is not None:
            zero = rows(zero)
    if zero is not None:
        codes = torch.stack((weight & 15, weight >> 4), dim=-1).reshape(n, -1, 128)
        values = codes.float() - zero.unsqueeze(-1)
    else:
        values = weight.reshape(n, -1, 128).float()
    return (values * scale.unsqueeze(-1)).reshape(n, k).to(dtype)


def affine_g128(
    q,
    scales,
    sums,
    weight,
    wscale,
    zero=None,
    *,
    plan=None,
    topk=1,
    down=False,
    split=1,
    dot8=False,
    wave_format=False,
    waves=2,
    triton_tiles=None,
    route_tile=16,
    packet_load=False,
    gated_quant=False,
    clamp_limit=None,
    route_group=0,
    output_dtype=torch.float32,
    symmetric_w4=False,
):
    """G128 W4A8 or symmetric W8A8 with the same tiling and route scatter.

    Weights are [N,K/2] U4 or [N,K] S8, optionally prefixed by experts.
    A routed plan uses sorted blocks padded with route_count. route_group controls
    how many adjacent route blocks share a column tile before advancing columns.
    An unsplit Triton projection can store its FP32 accumulator directly as FP16.
    """
    if q.ndim != 2 or q.dtype != torch.int8:
        raise ValueError("Expected a matrix of signed INT8 activations")
    m, k = q.shape
    w4 = zero is not None
    if symmetric_w4 and (not w4 or triton_tiles is None):
        raise ValueError("A fixed W4 zero point requires the packed Triton GEMM")
    if k % 128 or not 1 <= split <= k // 128 or topk < 1:
        raise ValueError("Invalid G128 geometry, split or route count")
    if weight.ndim != (2 if plan is None else 3):
        raise ValueError(
            "Expected dense or expert-packed weights matching the route plan"
        )
    n = weight.shape[-2]
    if (
        waves not in (2, 4, 8)
        or wave_format not in (0, 1, 2, 3)
        or (wave_format and n % 16)
        or (wave_format == 3 and n % 64)
    ):
        raise ValueError("Invalid affine wave geometry")
    if weight.shape[-1] != k // (2 if w4 else 1) or weight.dtype != (
        torch.uint8 if w4 else torch.int8
    ):
        raise ValueError("Weight storage differs from the selected G128 codec")
    group_shape = (*weight.shape[:-1], k // 128)
    if wscale.shape != group_shape or wscale.dtype != torch.float16:
        raise ValueError("Expected an FP16 scale for each weight group")
    if scales.shape != (m, k // 128) or scales.dtype != torch.float32:
        raise ValueError("Expected one FP32 scale per activation group")
    if w4 and (
        zero.shape != group_shape
        or zero.dtype != torch.float16
        or (not symmetric_w4 and sums is None)
        or (sums is not None and (sums.shape != scales.shape or sums.dtype != torch.int32))
    ):
        raise ValueError(
            "Affine INT4 requires floating zeros and INT32 activation sums"
        )
    if wave_format == 3 and triton_tiles is None and not dot8:
        triton_tiles = (route_tile if plan is not None else 32, 64, 4)
        packet_load = True
    if dot8 and (not w4 or plan is not None or split != 1):
        raise ValueError("Packed dot8 requires a dense, unsplit INT4 matrix")
    if plan is not None and split != 1:
        raise ValueError("Routed affine GEMM requires split=1")
    rows = m if plan is None or down else m * topk
    if output_dtype != torch.float32 and (
        output_dtype != torch.float16
        or triton_tiles is None
        or split != 1
        or gated_quant
    ):
        raise ValueError("Direct FP16 output requires an unsplit Triton GEMM")
    if gated_quant:
        if (
            triton_tiles is None
            or not wave_format
            or not packet_load
            or split != 1
            or n % 256
        ):
            raise ValueError(
                "Fused SwiGLU quantization requires paired packed G128 output tiles"
            )
        out = torch.empty((rows, n // 2), device=q.device, dtype=torch.int8)
        output_scales = torch.empty(
            (rows, n // 256), device=q.device, dtype=torch.float32
        )
        output_sums = torch.empty_like(output_scales, dtype=torch.int32)
    else:
        out = torch.empty((split, rows, n), device=q.device, dtype=output_dtype)
        output_scales = output_sums = None
    sorted_ids, experts, count = (None, None, None) if plan is None else plan
    tile_m = route_tile if triton_tiles is None else triton_tiles[0]
    if triton_tiles is None and (
        tile_m not in (16, 32, 64) or (tile_m > 16 and wave_format != 2)
    ):
        raise ValueError("Multi-wave row tiles require vector-packed weights")
    if plan is not None and (
        count.numel() != 1
        or any(t.dtype != torch.int32 for t in plan)
        or sorted_ids.numel() < tile_m * experts.numel()
    ):
        raise ValueError("Invalid 16-route-block dispatch plan")
    data = (q, scales, sums, weight, wscale, zero, out, sorted_ids, experts, count)
    if not q.is_cuda or any(
        t.device != q.device or not t.is_contiguous() for t in data if t is not None
    ):
        raise ValueError("All affine GEMM tensors must be contiguous on the same GPU")
    if not rows or not n:
        return (out, output_scales, output_sums) if gated_quant else out[0]
    if triton_tiles is not None:
        import triton

        from aiter.ops.triton._triton_kernels.gemm.basic.gemm_packed_g128 import _gemm

        tm, tn, warps = triton_tiles
        if (
            tm not in (16, 32, 64, 128)
            or tn not in (16, 32, 64, 128)
            or warps not in (2, 4, 8)
        ):
            raise ValueError("Unsupported affine GEMM tile")
        tiles = experts.numel() if plan is not None else triton.cdiv(rows, tm)
        if gated_quant:
            tn = 256
        _gemm[(tiles, triton.cdiv(n, tn), split)](
            q,
            scales,
            weight,
            None,
            None,
            None,
            out,
            output_scales,
            rows,
            n,
            k,
            0,
            k,
            False,
            False,
            tm,
            "affine_w4" if w4 else "signed_w8",
            IDS=experts,
            TOPK=topk,
            ROUTED=plan is not None,
            DOWN=down,
            WSTRIDE=weight.stride(0),
            SORTED=sorted_ids,
            COUNT=count,
            GROUPED=plan is not None,
            SPLIT_K=split,
            WEIGHT_SCALES=wscale,
            CACHED_SCALES=True,
            ASUM=sums,
            WZERO=zero,
            SYMMETRIC_W4=symmetric_w4,
            AFFINE_WAVE=wave_format,
            BN=tn,
            PACKET_LOAD=packet_load,
            GATE_OUT=gated_quant,
            GATE_LIMIT=clamp_limit,
            YSUM=output_sums,
            ROUTE_GROUP_M=route_group,
            num_warps=warps,
            num_stages=1,
            matrix_instr_nonkdim=16,
            enable_fp_fusion=False,
        )
        if gated_quant:
            return out, output_scales, output_sums
        return out.sum(0) if split != 1 else out[0]
    module, _ = _load()
    fn = module.launch_affine
    fn.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 11 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    status = fn(
        *(t.data_ptr() if t is not None else None for t in data),
        rows,
        n,
        k,
        experts.numel() if plan is not None else 0,
        topk,
        int(down),
        split,
        int(dot8),
        int(wave_format),
        waves,
        tile_m,
        torch.cuda.current_stream(q.device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"Affine G128 HIP launch failed: {status}")
    return out.sum(0) if split != 1 else out[0]
