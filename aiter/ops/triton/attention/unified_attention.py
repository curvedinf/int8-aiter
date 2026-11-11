# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import inspect
import math
import os

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    kernel_unified_attention_2d,
    kernel_unified_attention_3d,
    reduce_segments,
)
from aiter.ops.triton.utils.device_info import get_num_sms

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_3d import (
        _unified_attention_gluon_kernel_3d,
    )
except:  # noqa: E722
    _unified_attention_gluon_kernel_3d = None

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_2d import (
        _unified_attention_gluon_kernel_2d,
    )
except:  # noqa: E722
    _unified_attention_gluon_kernel_2d = None

try:
    from aiter.ops.triton._gluon_kernels.gfx1250.attention.unified_attention_reduce import (
        reduce_segments_gluon as _reduce_segments_gluon,
    )
except:  # noqa: E722
    _reduce_segments_gluon = None

from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import get_arch
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype

from aiter.ops.kv_rotation import (
    BLOCK as _ROT_BLOCK,
    enabled as _rot_enabled,
    inverse_ as _rot_inverse,
    rotate as _rot_forward,
)

# Max NUM_SEGMENTS the gluon reduce holds in-thread; larger split counts fall back to the Triton reduce_segments.
_GLUON_REDUCE_MAX_SEGMENTS = 8

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)
WARP_SIZE = 32 if IS_DEVICE_ARCH_GFX12 else 64
WAPR_SIZE_LOG2 = int(math.log2(WARP_SIZE))


def is_2d_gluon_available(
    q_dtype, kv_cache_dtype, softcap, use_qq_bias, use_alibi_slopes
):
    use_gluon_2d = (
        IS_DEVICE_ARCH_GFX12
        and _unified_attention_gluon_kernel_2d is not None
        and not softcap
        and not use_qq_bias
        and not use_alibi_slopes
        and q_dtype != torch.uint8
        and kv_cache_dtype != torch.uint8
        and q_dtype == kv_cache_dtype
    )
    return use_gluon_2d


def select_2d_config(
    block_size,
    head_size,
    sliding_window,
    all_decode,
    max_seqlen_q,
    max_seqlen_k,
    num_queries_per_kv,
    num_2d_prgms,
    q_dtype,
    kv_cache_dtype,
    shuffled_kv_cache,
):
    arch = get_arch()

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )

    TILE_SIZE = 32 if arch.name in ("gfx1201", "gfx908") else 16 if arch.is_rdna else 64
    waves_per_eu = 8 if arch.name == "gfx1151" else 6 if arch.is_rdna else 2

    # gfx908 int8 prefill: microbench shows waves_per_eu=1 is ~4.7x faster
    # than 2 for single-seq 5000-token prefill with int8 KV cache.
    if arch.name == "gfx908":
        waves_per_eu = 1

    max_num_stages_2d = 2 if head_size > 128 else 4

    # base prefill, for short cases
    if not all_decode:
        if head_size >= 512 and not arch.is_rdna:
            num_warps, num_stages_2d = 4, 2
            TILE_SIZE = 16
        elif head_size >= 256 and not arch.is_rdna:
            num_warps, num_stages_2d = 2, 2
            TILE_SIZE = 32
        else:
            # large prefill config
            if max_seqlen_q >= 256:
                BLOCK_M = 64 if arch.is_rdna else 128
                num_stages_2d, num_warps = 1, 4
            else:
                num_stages_2d, num_warps = 1, 2

    # pure decode config
    else:
        # to not have masking when loading KV
        _dec_tile = 64
        if os.environ.get("AITER_UA_DECODE_TILE") is not None:
            try:
                _dec_tile = int(os.environ["AITER_UA_DECODE_TILE"])
                assert _dec_tile in (16, 32, 64, 128, 256)
            except (ValueError, AssertionError):
                print("[aiter] ignoring invalid AITER_UA_DECODE_TILE", flush=True)
                _dec_tile = 64
        TILE_SIZE = min(_dec_tile, triton.next_power_of_2(block_size))
        if arch.is_rdna:
            num_stages_2d, num_warps = 1, 4
        else:
            if head_size >= 512:
                num_stages_2d, num_warps = 1, 4
            else:
                num_stages_2d, num_warps = 3, 2
            if os.environ.get("AITER_UA_DECODE_WARPS") is not None:
                try:
                    _w = int(os.environ["AITER_UA_DECODE_WARPS"])
                    _s = int(os.environ["AITER_UA_DECODE_STAGES"])
                    assert 1 <= _w <= 16 and 1 <= _s <= 4
                    num_warps, num_stages_2d = _w, _s
                except (ValueError, AssertionError, KeyError):
                    pass
        # G1 cross-path unification (AITER_UA_PIN_TILE=1): decode otherwise
        # uses TILE_SIZE 64 while prefill uses 32 (gfx908), so the softmax
        # accumulation order over KV tiles differs between decode and
        # prefill for the same token — a residual ~1e-4-class seed that
        # amplifies at 20k context (vllm-gfx908 ledger G1_* series). Pin
        # the decode tile/stages to the prefill values so both paths
        # accumulate in the same order.
        import os as _os

        if _os.environ.get("AITER_UA_PIN_TILE", "0") == "1":
            TILE_SIZE = 32
            num_stages_2d, num_warps = 2, 2

    BLOCK_Q = BLOCK_M // num_queries_per_kv
    num_stages_2d = min(max_num_stages_2d, num_stages_2d)

    # fix TILE_SIZE to block_size if shuffled_kv_cache is True
    if shuffled_kv_cache:
        if q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
            assert (
                block_size >= 32
            ), "For A8W8 Unified Attention with pre-shuffled KV cache, only block_size >= 32 is supported"
        TILE_SIZE = block_size
    elif q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
        TILE_SIZE = max(32, TILE_SIZE)

    return {
        "BLOCK_M": BLOCK_M,
        "BLOCK_Q": BLOCK_Q,
        "TILE_SIZE": TILE_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages_2d,
        "waves_per_eu": waves_per_eu,
    }


def select_3d_config(
    head_size,
    block_size,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
    q_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    shuffled_kv_cache: bool = False,
    NUM_BLOCKS_GATHER_PER_TILE: int = 1,
    SLIDING_WINDOW: int | None = None,
):
    arch = get_arch()
    reduce_num_warps = 2
    attn_warps = 2
    waves_per_eu = 2
    num_segments = 0
    attn_stages = 2
    if IS_DEVICE_ARCH_GFX12:
        assert kv_cache_dtype in (
            torch.float16,
            torch.bfloat16,
            e4m3_dtype,
            torch.uint8,
        ), f"kv_cache_dtype only supports F16 ({torch.float16}) BF16 ({torch.bfloat16}), FP8 ({e4m3_dtype}), FP4 ({torch.uint8}) in arch = {DEVICE_ARCH}"
        attn_warps = 1
        TILE_SIZE = block_size
        if shuffled_kv_cache and head_size < 128:
            if kv_cache_dtype in (
                torch.bfloat16,
                torch.float16,
            ):
                if block_size <= 64:
                    waves_per_eu = 2
                else:
                    waves_per_eu = 1
            elif kv_cache_dtype == e4m3_dtype:
                if block_size <= 128:
                    waves_per_eu = 2
                else:
                    waves_per_eu = 1
            else:
                assert block_size == 128, "FP4 KV cache only supports block_size 128"
                waves_per_eu = 2
        else:
            # GFX12 fallback
            waves_per_eu = 1

        if SLIDING_WINDOW is not None and SLIDING_WINDOW > 0:
            num_segments = 1
        else:
            occ = waves_per_eu * 4 // attn_warps
            MAX_SEGMENTS = max(1, math.ceil(max_seqlen_k / TILE_SIZE))
            num_segments = max(1, target_num_prgms // 4 * occ // max(1, num_2d_prgms))
            num_segments = min(MAX_SEGMENTS, num_segments)
            num_segments = triton.next_power_of_2(num_segments)

        # # this section increases the num_warps if the occ is too high
        # total_num_wg = num_2d_prgms * num_segments
        # if total_num_wg < occ * target_num_prgms:
        #     # occ too high, increase attn_warps to relax occ
        #     attn_warps = (waves_per_eu * 4) // max(
        #         1, triton.next_power_of_2(total_num_wg // target_num_prgms)
        #     )
        #     attn_warps = max(attn_warps, 1)
        #     attn_warps = min(attn_warps, 4)
    else:
        assert kv_cache_dtype in (
            torch.float16,
            torch.bfloat16,
            torch.int8,
            e4m3_dtype,
        ), f"kv_cache_dtype only supports F16 ({torch.float16}) BF16 ({torch.bfloat16}), INT8 ({torch.int8}), FP8 ({e4m3_dtype}) in arch = {DEVICE_ARCH}"

        if head_size >= 512 and not arch.is_rdna:
            attn_warps, attn_stages = 4, 1

        # gfx908 (MI100) tuning: lower waves_per_eu empirically improves
        # decode latency by issuing more independent thread groups across
        # the long-KV attention dimension, and num_warps=4 + num_stages=1
        # is 1.29x faster than the 2/2 default for int8 KV decode.
        if arch.name == "gfx908":
            waves_per_eu = 1
            attn_warps = 4
            attn_stages = 1
            # Experiment levers for the long-KV decode path (defaults keep
            # the measured short-context tuning).
            _w = os.environ.get("AITER_UA_3D_WARPS")
            _st = os.environ.get("AITER_UA_3D_STAGES")
            if _w:
                attn_warps = int(_w)
            if _st:
                attn_stages = int(_st)
        occ = waves_per_eu * 4 // attn_warps
        target_num_prgms = target_num_prgms * occ
        # Experiment lever: force the KV segment count for decode. The
        # heuristic below can leave the decode grid at a handful of CTAs at
        # long context (measured 2026-11-11: 881 us/layer at 16k, 12 seqs,
        # ~0.5% of peak); this lets the segment count be swept without a
        # rebuild of the heuristic.
        _seg = os.environ.get("AITER_UA_SEGMENTS")

        TILE_SIZE = min(64, triton.next_power_of_2(block_size))
        _tile = os.environ.get("AITER_UA_TILE")
        if _tile:
            TILE_SIZE = int(_tile)
            NUM_BLOCKS_GATHER_PER_TILE = max(1, TILE_SIZE // block_size)

        MAX_SEGMENTS = min(128, math.ceil(max_seqlen_k / TILE_SIZE))
        MIN_SEGMENTS = min(8, MAX_SEGMENTS)
        if head_size >= 512 and not arch.is_rdna:
            MIN_SEGMENTS = min(16, MAX_SEGMENTS)
        if num_segments == 0:
            num_segments = math.ceil(target_num_prgms / num_2d_prgms)
            num_segments = min(num_segments, MAX_SEGMENTS)
            num_segments = max(num_segments, MIN_SEGMENTS)
            num_segments = triton.next_power_of_2(num_segments)
        if _seg:
            num_segments = max(1, min(int(_seg), MAX_SEGMENTS))

        if num_segments == MIN_SEGMENTS:
            reduce_num_warps = 1

        if shuffled_kv_cache:
            if q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
                assert (
                    block_size >= 32
                ), "For A8W8 Unified Attention with pre-shuffled KV cache, only block_size >= 32 is supported"
            TILE_SIZE = block_size
        elif q_dtype == e4m3_dtype and kv_cache_dtype == e4m3_dtype:
            TILE_SIZE = max(32, TILE_SIZE)

    if NUM_BLOCKS_GATHER_PER_TILE > 1:
        # force gather mode
        assert NUM_BLOCKS_GATHER_PER_TILE in [
            2,
            4,
            8,
        ], "Only NUM_BLOCKS_GATHER_PER_TILE = 2, 4 or 8 is supported"
        attn_warps = 2
        waves_per_eu = 1
        num_segments = max(1, num_segments // NUM_BLOCKS_GATHER_PER_TILE)
        TILE_SIZE = block_size * NUM_BLOCKS_GATHER_PER_TILE
    elif TILE_SIZE > block_size:
        assert (
            TILE_SIZE % block_size == 0
        ), "TILE_SIZE needs to be divisible by block_size"
        NUM_BLOCKS_GATHER_PER_TILE = TILE_SIZE // block_size

    # gfx1151 (RDNA3.5) decode is memory-latency-bound at bs=1: the default 2
    # warps/workgroup leave unified_attention at only ~31% of the LPDDR5X
    # bandwidth roofline. 8 warps/workgroup reach ~59% (1.5-1.9x on bf16 decode)
    # with bitwise-identical output. Mirrors the waves_per_eu=8 gfx1151 tuning above.
    if DEVICE_ARCH == "gfx1151":
        attn_warps = 8

    attn_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": attn_warps,
        "waves_per_eu": waves_per_eu,
        "num_stages": attn_stages,
    }

    reduce_config = {
        "TILE_SIZE": TILE_SIZE,
        "NUM_SEGMENTS_PER_SEQ": num_segments,
        "num_warps": reduce_num_warps,
        "waves_per_eu": 2,
        "num_stages": 1,
    }

    return attn_config, reduce_config


def use_2d_kernel(
    head_size,
    sliding_window,
    all_decode,
    max_seqlen_q,
    max_seqlen_k,
    target_num_prgms,
    num_2d_prgms,
):
    # G1 cross-path unification (AITER_UA_FORCE_2D=1): decode batches select
    # the 3D/flash-decoding kernel while prefill uses 2D, and the two kernel
    # families differ by ~6e-5 on identical inputs — at 20k context that
    # residual is amplified into decode-vs-prefill distribution divergence
    # (vllm-gfx908 ledger G1_RESIDUAL_SEED_NARROWED_GDN_PREFILL onward).
    # Forcing 2D for every batch makes decode and prefill share one kernel
    # path (2D per-row math is batch-invariant). Decode gets slower; this is
    # a correctness lever first, speed recovery later.
    import os as _os

    if _os.environ.get("AITER_UA_FORCE_2D", "0") == "1":
        return True

    # if IS_DEVICE_ARCH_GFX12, always use 3D if all_decode and 2D otherwise
    if IS_DEVICE_ARCH_GFX12:
        return (sliding_window > 0) or (not all_decode)

    if head_size >= 512 and not get_arch().is_rdna and not all_decode:
        return True

    return (
        (sliding_window > 0)
        or (max_seqlen_k <= 512)
        or (num_2d_prgms > target_num_prgms)
    )


def _unified_attention_impl(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    shuffled_kv_cache: bool = False,
    skip_reduce: bool = False,
    # Per-token-head K/V scales for int8/fp8 per-token-head KV caches.
    # Shape: [num_blocks, block_size, num_kv_heads], dtype float32.
    k_scale_cache=None,
    v_scale_cache=None,
):
    assert causal, "Only causal attention is supported"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    use_per_token_head_scales = (
        k_scale_cache is not None and v_scale_cache is not None
    )
    if use_per_token_head_scales:
        assert not shuffled_kv_cache, (
            "Per-token-head KV scales are only supported for non-shuffled "
            "flash layout in this version."
        )
    SLIDING_WINDOW = 1 + window_size[0]

    q_dtype = q.dtype
    kv_cache_dtype = k.dtype
    use_int8_qk_dot = use_per_token_head_scales and kv_cache_dtype == torch.int8
    num_tokens, num_query_heads, head_size = q.shape

    if sinks is not None:
        assert sinks.shape[0] == num_query_heads, "Sinks must be num_query_heads size"

    BLOCK_SCALES_SIZE = 16
    if q_dtype == torch.uint8:
        # A4W4
        assert q_scales is not None and q_scales.dtype == e4m3_dtype
        head_size = head_size * 2
        QUERY_DTYPE = "nvfp4"
    elif q_dtype == e4m3_dtype:
        QUERY_DTYPE = "fp8"
    else:
        QUERY_DTYPE = "bf16"

    if kv_cache_dtype == torch.uint8:
        KV_CACHE_DTYPE = "nvfp4"
    elif kv_cache_dtype == e4m3_dtype:
        KV_CACHE_DTYPE = "fp8"
    elif kv_cache_dtype == torch.int8:
        KV_CACHE_DTYPE = "int8"
    else:
        KV_CACHE_DTYPE = "bf16"

    if shuffled_kv_cache:
        SCALE_K_WIDTH = 4
        if kv_cache_dtype == torch.uint8:
            num_blocks, num_kv_heads, block_size, _ = k.shape
            K_WIDTH = 16
            SCALE_K = head_size // 16
            SCALE_K_WIDTH = (
                min(16, triton.next_power_of_2(SCALE_K)) if SCALE_K >= 4 else SCALE_K
            )
        else:
            # key_cache: num_blocks, num_kv_heads, head_size // x, block_size, x
            # value_cache: num_blocks, num_kv_heads, block_size // x, head_size, x
            num_blocks, num_kv_heads, _, block_size, K_WIDTH = k.shape
    else:
        # key_cache and value_cache: num_blocks, block_size, num_kv_heads, head_size
        num_blocks, block_size, num_kv_heads, _ = k.shape
        K_WIDTH = 16 if kv_cache_dtype == e4m3_dtype else 8
        SCALE_K_WIDTH = 4

    num_seqs = len(seqused_k)
    num_queries_per_kv = num_query_heads // num_kv_heads

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    # gfx908 int8 decode verify: pack four verify positions into one M tile
    # (4 x 8 q heads = 32 rows) so each KV segment is streamed once per four
    # positions instead of once per two. Microbench, 16k context, kvh1:
    # BLOCK_M 16 -> 506.8 us/layer, 32 -> 393.8, 64 -> 442.1 (the PV
    # accumulator at 64 is 128 f32/lane and spills).
    if DEVICE_ARCH == "gfx908" and max_seqlen_q <= 16 and num_queries_per_kv == 8:
        BLOCK_M = 32
    # Native gfx908 int8 decode path (docs/design/int8_small_gemm_native.md,
    # stage 1). Fires only for the exact contract the HIP kernel implements:
    # gfx908, int8 per-token-head cache, head_dim 128, causal, verify block
    # <= 8, GQA 8 or 4. Everything else falls through to the Triton kernels.
    if (
        max_seqlen_q <= 8
        and window_size[0] < 0  # native kernel has no sliding window
        and softcap == 0
        and alibi_slopes is None
        and qq_bias is None
        and output_scale is None
        and num_queries_per_kv in (4, 8)
        and head_size == 128
        and kv_cache_dtype == torch.int8
        and k_scale_cache is not None
        and v_scale_cache is not None
        and q.dtype == torch.bfloat16
        and q.is_contiguous()
        and out.is_contiguous()
    ):
        from aiter.ops import int8_native as _i8n

        if _i8n.native_enabled("ua_decode") and not _BYPASS_NATIVE:
            if os.environ.get("INT8_ATTN_TRACE") == "1":
                _key = (tuple(q.shape), num_queries_per_kv, max_seqlen_q,
                        max_seqlen_k // 512, window_size[0], sinks is not None)
                if _key not in _NATIVE_TRACE:
                    _NATIVE_TRACE.add(_key)
                    print(f"[int8-native] q={tuple(q.shape)} qg={num_queries_per_kv} "
                          f"maxq={max_seqlen_q} maxk={max_seqlen_k} "
                          f"win={window_size[0]} sinks={sinks is not None}",
                          flush=True)
            _seg = 8 if max_seqlen_k <= 4096 else 16
            _seg = min(_seg, max(1, (max_seqlen_k + 63) // 64))
            _i8n.paged_attn_decode_int8(
                q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k,
                block_table, k_scale_cache, v_scale_cache, softmax_scale,
                num_segments=_seg, sinks=sinks,
            )
            return out

    # Experiment lever: the verify block is num_queries_per_kv queries; the
    # 16-row floor pads it 2x when nqpk=8 and doubles the dot work.
    _bm = os.environ.get("AITER_UA_BLOCK_M")
    if _bm:
        BLOCK_M = int(_bm)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1
    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    cu_count = get_num_sms()
    target_num_prgms = cu_count * 4
    ALL_DECODE = max_seqlen_q == 1
    if ALL_DECODE:
        total_num_q_blocks = num_seqs
    else:
        total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    ALL_DECODE = int(max_seqlen_q) == 1
    # if batch contains a prefill
    if use_2d_kernel(
        head_size,
        SLIDING_WINDOW,
        ALL_DECODE,
        max_seqlen_q,
        max_seqlen_k,
        target_num_prgms,
        num_2d_prgms,
    ):

        # The gfx1250 Gluon 2d kernel only handles bf16/fp8 q+kv (with optional
        # sinks / output_scale / shuffled_kv_cache)
        use_gluon_2d = is_2d_gluon_available(
            q_dtype, kv_cache_dtype, softcap, use_qq_bias, use_alibi_slopes
        )
        if use_gluon_2d:
            _gfx1250_unified_attention_2d(
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                seqused_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                causal,
                window_size,
                block_table,
                softcap,
                q_descale,
                k_descale,
                v_descale,
                sinks,
                output_scale=output_scale,
                shuffled_kv_cache=shuffled_kv_cache,
            )
        else:
            config = select_2d_config(
                block_size,
                head_size,
                SLIDING_WINDOW,
                ALL_DECODE,
                max_seqlen_q,
                max_seqlen_k,
                num_queries_per_kv,
                num_2d_prgms,
                q_dtype,
                kv_cache_dtype,
                shuffled_kv_cache,
            )
            assert config["BLOCK_Q"] >= 1
            if ALL_DECODE:
                total_num_q_blocks = num_seqs
            else:
                total_num_q_blocks = q.shape[0] // config["BLOCK_Q"] + num_seqs

            kernel_unified_attention_2d[
                (
                    num_kv_heads,
                    total_num_q_blocks,
                )
            ](
                output_ptr=out,
                query_ptr=q,
                key_cache_ptr=k,
                value_cache_ptr=v,
                sink_ptr=sinks,
                block_tables_ptr=block_table,
                seq_lens_ptr=seqused_k,
                alibi_slopes_ptr=alibi_slopes,
                qq_bias_ptr=qq_bias,
                scale=softmax_scale,
                q_descale_ptr=q_descale,
                k_descale_ptr=k_descale,
                v_descale_ptr=v_descale,
                out_scale_ptr=output_scale,
                softcap=softcap,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                block_table_stride=block_table.stride(0),
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                output_stride_0=out.stride(0),
                output_stride_1=out.stride(1),
                qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
                BLOCK_SIZE=block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                USE_QQ_BIAS=use_qq_bias,
                USE_SOFTCAP=(softcap > 0),
                USE_SINKS=(sinks is not None),
                SLIDING_WINDOW=SLIDING_WINDOW,
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                query_start_len_ptr=cu_seqlens_q,
                k_scale_cache_ptr=k_scale_cache if use_per_token_head_scales else None,
                v_scale_cache_ptr=v_scale_cache if use_per_token_head_scales else None,
                stride_ks_blk=k_scale_cache.stride(0) if use_per_token_head_scales else 0,
                stride_ks_slot=k_scale_cache.stride(1) if use_per_token_head_scales else 0,
                stride_ks_head=k_scale_cache.stride(2) if use_per_token_head_scales else 0,
                stride_vs_blk=v_scale_cache.stride(0) if use_per_token_head_scales else 0,
                stride_vs_slot=v_scale_cache.stride(1) if use_per_token_head_scales else 0,
                stride_vs_head=v_scale_cache.stride(2) if use_per_token_head_scales else 0,
                num_seqs=num_seqs,
                ALL_DECODE=ALL_DECODE,
                SHUFFLED_KV_CACHE=shuffled_kv_cache,
                K_WIDTH=K_WIDTH,
                USE_PER_TOKEN_HEAD_SCALES=use_per_token_head_scales,
                USE_INT8_QK_DOT=use_int8_qk_dot,
                ABL=int(os.environ.get("AITER_UA_ABL", "0")),
                **config,
            )
        return out

    else:
        NUM_BLOCKS_GATHER_PER_TILE = 1
        attn_config, reduce_config = select_3d_config(
            head_size,
            block_size,
            max_seqlen_k,
            target_num_prgms,
            num_2d_prgms,
            q_dtype,
            kv_cache_dtype,
            shuffled_kv_cache,
            NUM_BLOCKS_GATHER_PER_TILE,
            SLIDING_WINDOW,
        )
        # gfx908 decode tuning (measured 2026-11-11, 12 seqs x 8 q):
        # warps=2 / stages=2 / 8 segments is 1.8x faster than the previous
        # warps=4 / stages=1 / heuristic-segments geometry at 16k context
        # (487 vs 881 us/layer) and 1.5x at 2k (74 vs 109), and 1.6-2.1x
        # faster than bf16 KV on the same kernel. Decode-only: prefill keeps
        # its own tuning. Env levers keep the sweep reproducible.
        if DEVICE_ARCH == "gfx908" and max_seqlen_q <= 16:
            _seg = int(os.environ.get("AITER_UA_SEGMENTS", "8"))
            attn_config["NUM_SEGMENTS_PER_SEQ"] = _seg
            reduce_config["NUM_SEGMENTS_PER_SEQ"] = _seg
            attn_config["num_warps"] = int(os.environ.get("AITER_UA_3D_WARPS", "2"))
            attn_config["num_stages"] = int(os.environ.get("AITER_UA_3D_STAGES", "2"))
        NUM_SEGMENTS = attn_config["NUM_SEGMENTS_PER_SEQ"]
        if NUM_SEGMENTS > 1:
            segm_output = torch.empty(
                q.shape[0],
                num_query_heads,
                NUM_SEGMENTS,
                triton.next_power_of_2(head_size),
                dtype=torch.float32,
                device=q.device,
            )
            segm_max = torch.empty(
                q.shape[0],
                num_query_heads,
                NUM_SEGMENTS,
                dtype=torch.float32,
                device=q.device,
            )
            segm_expsum = torch.empty(
                q.shape[0],
                num_query_heads,
                NUM_SEGMENTS,
                dtype=torch.float32,
                device=q.device,
            )
        else:
            segm_output = out
            segm_max = out  # dummy ptr
            segm_expsum = out  # dummy ptr

        if IS_DEVICE_ARCH_GFX12 and shuffled_kv_cache:
            _unified_attention_gluon_kernel_3d[
                (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)
            ](
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                query_ptr=q,
                query_scales_ptr=q_scales,
                key_cache_ptr=k,
                value_cache_ptr=v,
                sink_ptr=sinks,
                block_tables_ptr=block_table,
                seq_lens_ptr=seqused_k,
                alibi_slopes_ptr=alibi_slopes,
                qq_bias_ptr=qq_bias,
                q_scale_ptr=q_descale,
                k_scale_ptr=k_descale,
                v_scale_ptr=v_descale,
                out_scale_ptr=(
                    output_scale
                    if (output_scale is not None and NUM_SEGMENTS == 1)
                    else None
                ),
                softcap=softcap,
                num_seqs=num_seqs,
                num_blocks=num_blocks,
                block_table_stride=block_table.stride(0),
                max_num_blocks_per_seq=block_table.shape[1],
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                query_scales_stride_0=q_scales.stride(0) if q_scales is not None else 0,
                query_scales_stride_1=q_scales.stride(1) if q_scales is not None else 0,
                qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
                BLOCK_SIZE=block_size,
                HEAD_SIZE=head_size,
                USE_ALIBI_SLOPES=use_alibi_slopes,
                USE_QQ_BIAS=use_qq_bias,
                USE_SOFTCAP=(softcap > 0),
                USE_SINKS=(sinks is not None),
                SLIDING_WINDOW=SLIDING_WINDOW,
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                query_start_len_ptr=cu_seqlens_q,
                SCALE=softmax_scale,
                NUM_QUERY_HEADS=num_query_heads,
                NUM_KV_HEADS=num_kv_heads,
                BLOCK_Q=BLOCK_Q,
                BLOCK_M=BLOCK_M,
                ALL_DECODE=ALL_DECODE,
                SHUFFLED_KV_CACHE=shuffled_kv_cache,
                K_WIDTH=K_WIDTH,
                SCALE_K_WIDTH=SCALE_K_WIDTH,
                WARP_SIZE=WARP_SIZE,
                NUM_BLOCKS_GATHER_PER_TILE=NUM_BLOCKS_GATHER_PER_TILE,
                QUERY_DTYPE=QUERY_DTYPE,
                KV_CACHE_DTYPE=KV_CACHE_DTYPE,
                BLOCK_SCALES_SIZE=BLOCK_SCALES_SIZE,
                **attn_config,
            )
        else:
            kernel_unified_attention_3d[
                (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)
            ](
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                query_ptr=q,
                key_cache_ptr=k,
                value_cache_ptr=v,
                sink_ptr=sinks,
                block_tables_ptr=block_table,
                seq_lens_ptr=seqused_k,
                alibi_slopes_ptr=alibi_slopes,
                qq_bias_ptr=qq_bias,
                scale=softmax_scale,
                q_descale_ptr=q_descale,
                k_descale_ptr=k_descale,
                v_descale_ptr=v_descale,
                out_scale_ptr=(
                    output_scale
                    if (output_scale is not None and NUM_SEGMENTS == 1)
                    else None
                ),
                softcap=softcap,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                block_table_stride=block_table.stride(0),
                query_stride_0=q.stride(0),
                query_stride_1=q.stride(1),
                qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
                BLOCK_SIZE=block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                USE_QQ_BIAS=use_qq_bias,
                USE_SOFTCAP=(softcap > 0),
                USE_SINKS=(sinks is not None),
                SLIDING_WINDOW=SLIDING_WINDOW,
                stride_k_cache_0=k.stride(0),
                stride_k_cache_1=k.stride(1),
                stride_k_cache_2=k.stride(2),
                stride_k_cache_3=k.stride(3),
                stride_v_cache_0=v.stride(0),
                stride_v_cache_1=v.stride(1),
                stride_v_cache_2=v.stride(2),
                stride_v_cache_3=v.stride(3),
                query_start_len_ptr=cu_seqlens_q,
                k_scale_cache_ptr=k_scale_cache if use_per_token_head_scales else None,
                v_scale_cache_ptr=v_scale_cache if use_per_token_head_scales else None,
                stride_ks_blk=k_scale_cache.stride(0) if use_per_token_head_scales else 0,
                stride_ks_slot=k_scale_cache.stride(1) if use_per_token_head_scales else 0,
                stride_ks_head=k_scale_cache.stride(2) if use_per_token_head_scales else 0,
                stride_vs_blk=v_scale_cache.stride(0) if use_per_token_head_scales else 0,
                stride_vs_slot=v_scale_cache.stride(1) if use_per_token_head_scales else 0,
                stride_vs_head=v_scale_cache.stride(2) if use_per_token_head_scales else 0,
                BLOCK_Q=BLOCK_Q,
                num_seqs=num_seqs,
                BLOCK_M=BLOCK_M,
                ALL_DECODE=ALL_DECODE,
                SHUFFLED_KV_CACHE=shuffled_kv_cache,
                K_WIDTH=K_WIDTH,
                IS_Q_FP8=(q_dtype == e4m3_dtype),
                IS_KV_FP8=(kv_cache_dtype == e4m3_dtype),
                USE_PER_TOKEN_HEAD_SCALES=use_per_token_head_scales,
                USE_INT8_QK_DOT=use_int8_qk_dot,
                ABL=int(os.environ.get("AITER_UA_ABL", "0")),
                LOOP_STAGES=int(os.environ.get("AITER_UA_LOOPSTAGES", "1")),
                **attn_config,
            )

        if NUM_SEGMENTS == 1:
            return segm_output
        elif skip_reduce:
            return segm_output, segm_max, segm_expsum

        head_size_padded = triton.next_power_of_2(head_size)
        # Gluon reduce (one workgroup/token, in-wave segment merge); valid for all-decode with small split counts, else the Triton reduce_segments.
        gluon_num_warps = 8 if num_query_heads % 8 == 0 else 4
        use_gluon_reduce = (
            IS_DEVICE_ARCH_GFX12
            and _reduce_segments_gluon is not None
            and ALL_DECODE
            and NUM_SEGMENTS <= _GLUON_REDUCE_MAX_SEGMENTS
            and head_size_padded % 32 == 0
            and num_query_heads % gluon_num_warps == 0
        )
        if use_gluon_reduce:
            _reduce_segments_gluon[(q.shape[0],)](
                output_ptr=out,
                segm_output_ptr=segm_output,
                segm_max_ptr=segm_max,
                segm_expsum_ptr=segm_expsum,
                seq_lens_ptr=seqused_k,
                num_query_heads=num_query_heads,
                out_scale_ptr=output_scale,
                output_stride_0=out.stride(0),
                output_stride_1=out.stride(1),
                H=num_query_heads,
                S=NUM_SEGMENTS,
                D=head_size,
                D_PAD=head_size_padded,
                TILE_SIZE=reduce_config["TILE_SIZE"],
                NUM_WARPS=gluon_num_warps,
                IS_FP8_OUT=(out.dtype == e4m3_dtype),
                FP8_MIN=torch.finfo(e4m3_dtype).min,
                FP8_MAX=torch.finfo(e4m3_dtype).max,
                num_warps=gluon_num_warps,
            )
            return out

        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_ptr=output_scale,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            **reduce_config,
        )

        return out


def _gfx1250_unified_attention_2d(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    sinks,
    output_scale=None,
    shuffled_kv_cache=False,
    loop_variant=None,
):
    """
    Internal wrapper for the gfx1250 gluon kernel.

    Args:
        See main wrapper for other args.
        loop_variant:
            0=plain double buffered version,
            1=2-stage version,
            2=4-stage version
    """
    # useful for debugging when needed
    remove_indirect_access = False
    NUM_SEQS = len(seqused_k)
    NUM_Q_HEADS = q.shape[1]
    HEAD_SIZE = q.shape[2]
    num_blocks = k.shape[0]
    Q_FP8 = q.element_size() == 1
    KV_FP8 = k.element_size() == 1
    ARCH_NAME = arch_info.get_arch()
    assert loop_variant in [
        None,
        0,
        1,
        2,
    ], "Only [None, 0, 1, 2] supported as loop_variant"
    assert ARCH_NAME == "gfx1250", "unified_attention_2d_gfx1250 only supports gfx1250"
    assert softcap == 0, "Softcap is not supported"
    if shuffled_kv_cache:
        # key_cache: num_blocks, num_kv_heads, head_size // x, block_size, x
        # value_cache: num_blocks, num_kv_heads, block_size // x, head_size, x
        num_blocks, NUM_KV_HEADS, _, BLOCK_SIZE, _K_WIDTH = k.shape
    else:
        BLOCK_SIZE = k.shape[1]
        NUM_KV_HEADS = k.shape[2]

    SLIDING_WINDOW = 1 + window_size[0]
    ALL_DECODE = max_seqlen_q == 1
    NUM_QUERIES_PER_KV = NUM_Q_HEADS // NUM_KV_HEADS
    num_buffers = None
    if ALL_DECODE:
        sel_loop_variant = 0
        BLOCK_M = (
            16
            if NUM_QUERIES_PER_KV <= 16
            else triton.next_power_of_2(NUM_QUERIES_PER_KV)
        )
        num_warps = 1
        waves_per_eu = 2
        TILE_SIZE = 128 if (Q_FP8 and KV_FP8) else 64
    elif SLIDING_WINDOW > 0:
        # Prefill, sliding window
        sel_loop_variant = 0
        BLOCK_M = 64
        num_warps = 4
        waves_per_eu = 4
        TILE_SIZE = 128 if (Q_FP8 and KV_FP8) else 64
    else:
        # Prefill, full attention
        sel_loop_variant = 2
        BLOCK_M = 128
        num_warps = 4
        waves_per_eu = 2
        TILE_SIZE = 128 if (Q_FP8 and KV_FP8) else 64
        num_buffers = 2

        if max_seqlen_k < 2048:
            BLOCK_M = 64
            num_warps = 2 if (Q_FP8 and KV_FP8) else 1
            sel_loop_variant = 0
            num_buffers = 2

    loop_variant = sel_loop_variant if loop_variant is None else loop_variant
    # Non-shuffled KV can't use TDM gather (KV layout), so a tile is one page
    if not shuffled_kv_cache or TILE_SIZE < BLOCK_SIZE:
        TILE_SIZE = BLOCK_SIZE

    num_kv_blocks = TILE_SIZE // BLOCK_SIZE if shuffled_kv_cache else 1
    assert (
        num_kv_blocks & (num_kv_blocks - 1) == 0
    ), "num_kv_blocks must be a power of 2"

    assert (
        TILE_SIZE >= BLOCK_SIZE
    ), f"TILE_SIZE={TILE_SIZE} must be multiple of PAGE_SIZE={BLOCK_SIZE}"

    BLOCK_Q = BLOCK_M // NUM_QUERIES_PER_KV
    # Upper bound on masked tiles
    query_span = (BLOCK_M - 1) // NUM_QUERIES_PER_KV + 1
    max_mask_tiles = (query_span + TILE_SIZE - 1) // TILE_SIZE + 1
    # other variants do at most 2 masking at the end of loop
    if max_mask_tiles > 2:
        loop_variant = 0
    # fall back to the standard double-buffered loop
    if TILE_SIZE <= 32:
        loop_variant = 0
    if not ALL_DECODE:
        total_query_blocks = q.shape[0] // BLOCK_Q + NUM_SEQS
    else:
        total_query_blocks = NUM_SEQS
    NUM_WARPS = num_warps
    if num_buffers is None:
        num_buffers = 2 if loop_variant == 0 else 3
        num_buffers = 2 if ALL_DECODE else num_buffers

    kv_size = k.nelement() * k.element_size()
    MAX_INT32 = 2**31 - 1
    USE_LOAD_BUFFER_OP = ARCH_NAME != "gfx1250" and kv_size <= MAX_INT32
    USE_STORE_BUFFER_OP = out.nelement() * out.element_size() <= MAX_INT32
    grid = (NUM_KV_HEADS, total_query_blocks)
    _unified_attention_gluon_kernel_2d[grid](
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        output_ptr=out,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        query_start_len_ptr=cu_seqlens_q,
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        k_descale_ptr=k_descale,
        v_descale_ptr=v_descale,
        q_descale_ptr=q_descale,
        out_scale_ptr=output_scale,
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=SLIDING_WINDOW,
        num_blocks=num_blocks,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        block_table_stride=block_table.stride(0),
        num_seqs=NUM_SEQS,
        SCALE=softmax_scale,
        NUM_QUERY_HEADS=NUM_Q_HEADS,
        NUM_KV_HEADS=NUM_KV_HEADS,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        ARCH_NAME=ARCH_NAME,
        waves_per_eu=waves_per_eu,
        USE_LOAD_BUFFER_OP=USE_LOAD_BUFFER_OP,
        USE_STORE_BUFFER_OP=USE_STORE_BUFFER_OP,
        num_warps=NUM_WARPS,
        ALL_DECODE=ALL_DECODE,
        SHUFFLED_KV_CACHE=shuffled_kv_cache,
        CAUSAL=causal,
        REMOVE_INDIRECT_ACCESS=remove_indirect_access,
        NUM_BUFFERS=num_buffers,
        LOOP_VARIANT=loop_variant,
    )


# Public entry. Wraps the implementation so the int8 per-token-head KV
# Hadamard rotation (llama.cpp #21038) is applied exactly once around every
# attention call: Q rotated and softmax_scale divided by the block size on the
# way in, the output un-rotated on the way out. The cache writer rotates K/V
# with the same matrix, so scores are unchanged and the quantized cache is
# smoother. Gated to int8 per-token-head caches (k_scale_cache present).
_PARAM_NAMES = list(inspect.signature(_unified_attention_impl).parameters)
_NATIVE_TRACE: set = set()
_BYPASS_NATIVE = False


def _rot_arg(args, kwargs, name):
    i = _PARAM_NAMES.index(name)
    return args[i] if i < len(args) else kwargs.get(name)


def _rot_override(args, kwargs, values):
    args = list(args)
    for name, val in values.items():
        i = _PARAM_NAMES.index(name)
        if i < len(args):
            args[i] = val
        else:
            kwargs[name] = val
    return args, kwargs


def unified_attention(*args, **kwargs):
    q = _rot_arg(args, kwargs, "q")
    k = _rot_arg(args, kwargs, "k")
    out = _rot_arg(args, kwargs, "out")
    scale = _rot_arg(args, kwargs, "softmax_scale")
    k_scale_cache = _rot_arg(args, kwargs, "k_scale_cache")
    rot = (
        q is not None
        and out is not None
        and scale is not None
        and getattr(k, "dtype", None) == torch.int8
        and k_scale_cache is not None
        and q.shape[-1] % _ROT_BLOCK == 0
        and _rot_enabled(q.shape[-1])
    )
    if rot:
        args, kwargs = _rot_override(
            args,
            kwargs,
            {"q": _rot_forward(q)},
        )
    _cmp = (
        os.environ.get("INT8_ATTN_COMPARE") == "1"
        and not _BYPASS_NATIVE
        and _rot_arg(args, kwargs, "max_seqlen_q") is not None
        and _rot_arg(args, kwargs, "max_seqlen_q") <= 8
        and getattr(_rot_arg(args, kwargs, "k"), "dtype", None) == torch.int8
        and _rot_arg(args, kwargs, "k_scale_cache") is not None
    )
    _cmp_ref = None
    if _cmp:
        _cmp_ref = torch.empty_like(out)
        _ref_args, _ref_kwargs = _rot_override(
            args, kwargs, {"out": _cmp_ref}
        )
        globals()["_BYPASS_NATIVE"] = True
        try:
            _unified_attention_impl(*_ref_args, **_ref_kwargs)
        finally:
            globals()["_BYPASS_NATIVE"] = False
    result = _unified_attention_impl(*args, **kwargs)
    if _cmp and not isinstance(result, tuple):
        d = (_cmp_ref.float() - out.float()).abs()
        print(f"[int8-cmp] q={tuple(q.shape)} max={d.max().item():.4g} "
              f"mean={d.mean().item():.4g} ref_norm={_cmp_ref.float().norm().item():.4g} "
              f"nat_norm={out.float().norm().item():.4g}", flush=True)
        if not torch.cuda.is_current_stream_capturing():
            from aiter.ops import int8_native as _i8n
            _k = _rot_arg(args, kwargs, "k")
            _v = _rot_arg(args, kwargs, "v")
            _cu = _rot_arg(args, kwargs, "cu_seqlens_q")
            _sk = _rot_arg(args, kwargs, "seqused_k")
            _bt = _rot_arg(args, kwargs, "block_table")
            _ksc = _rot_arg(args, kwargs, "k_scale_cache")
            _vsc = _rot_arg(args, kwargs, "v_scale_cache")
            _sc = _rot_arg(args, kwargs, "softmax_scale")
            _dump = os.environ.get("INT8_ATTN_DUMP")
            if _dump and not os.path.exists(_dump):
                torch.save({
                    "q": q.detach().cpu(), "k": _k.detach().cpu(),
                    "v": _v.detach().cpu(), "cu": _cu.detach().cpu(),
                    "sk": _sk.detach().cpu(), "bt": _bt.detach().cpu(),
                    "ksc": _ksc.detach().cpu(), "vsc": _vsc.detach().cpu(),
                    "sinks": None if sinks is None else sinks.detach().cpu(),
                    "out_ref": _cmp_ref.detach().cpu(),
                    "out_nat": out.detach().cpu(),
                    "scale": _sc,
                }, _dump)
                print(f"[int8-dump] wrote {_dump}", flush=True)
            print(f"[int8-dbg] cu={_cu[:4].tolist()} sk={_sk[:4].tolist()} "
                  f"bt={_bt[0,:4].tolist()} dt={_cu.dtype},{_sk.dtype},{_bt.dtype} "
                  f"pre_norm={out.float().norm().item():.4g} "
                  f"qsc_head={float(q.float()[0].abs().max()):.4g}", flush=True)
            for _s in (1, 2, 4, 8, 16):
                _tmp = torch.empty_like(out)
                try:
                    _i8n.paged_attn_decode_int8(q, _k, _v, _tmp, _cu,
                        _rot_arg(args, kwargs, "max_seqlen_q"), _sk, _bt,
                        _ksc, _vsc, _sc, num_segments=_s, sinks=_rot_arg(args, kwargs, "sinks"))
                    _d = (_cmp_ref.float() - _tmp.float()).abs()
                    print(f"[int8-seg] seg={_s} max={_d.max().item():.4g} "
                          f"mean={_d.mean().item():.4g} norm={_tmp.float().norm().item():.4g}",
                          flush=True)
                except Exception as _e:
                    print(f"[int8-seg] seg={_s} FAIL {_e}", flush=True)
    # Tuple returns leave `out` to a later reduce (segmented/g8 path only,
    # never int8 per-token-head); only the final output is un-rotated.
    if rot and not isinstance(result, tuple):
        _rot_inverse(out)
    return result
