# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Native gfx908 RMSNorm with GS128 quantization of the stored output."""

import triton
import triton.language as tl
import triton.language.extra.libdevice as lib


@triton.jit
def _rms_norm_g128(
    input_ptr,
    output_ptr,
    g_ptr,
    rsigma_ptr,
    Q,
    S,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols: tl.constexpr,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
        input_ptrs = tl.multiple_of(input_ptrs, (16,))
        row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
            tl.float32
        )
        g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        row_norm = row * row
        row_norm = tl.sum(row_norm, axis=-1)
        norm_factor = tl.math.rsqrt(row_norm / n_cols + epsilon)
        tl.store(rsigma_ptr + row_idx, norm_factor)
        rms_norm = row * norm_factor * g
        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        output_ptrs = tl.multiple_of(output_ptrs, (16,))
        tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)
        v = rms_norm.to(output_ptr.type.element_ty).to(tl.float32)
        grouped = tl.reshape(v, (BLOCK_SIZE // 128, 128))
        scale = tl.maximum(tl.div_rn(tl.max(tl.abs(grouped), 1), 127.0), 1e-30)
        quant = lib.nearbyint(tl.div_rn(grouped, scale[:, None])).to(tl.int8)
        cols = tl.reshape(col_offsets, (BLOCK_SIZE // 128, 128))
        tl.store(Q + row_idx * n_cols + cols, quant)
        tl.store(S + row_idx * (n_cols // 128) + tl.arange(0, BLOCK_SIZE // 128), scale)


@triton.jit
def _fused_add_rms_norm_g128(
    input_ptr,
    output_ptr,
    res_in_ptr,
    res_out_ptr,
    g_ptr,
    rsigma_ptr,
    Q,
    S,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols: tl.constexpr,
    epsilon,
    BLOCK_SIZE: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
        input_ptrs = tl.multiple_of(input_ptrs, (16,))
        row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        res_in_ptrs = res_in_ptr + row_idx * input_row_stride + col_offsets
        res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
        res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        row += res_in
        res_out_ptrs = res_out_ptr + row_idx * input_row_stride + col_offsets
        res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
        tl.store(res_out_ptrs, row.to(res_out_ptr.type.element_ty), mask=mask)
        row = row.to(tl.float32)
        g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        row_norm = row * row
        row_norm = tl.sum(row_norm, axis=-1)
        norm_factor = tl.math.rsqrt(row_norm / n_cols + epsilon)
        tl.store(rsigma_ptr + row_idx, norm_factor)
        rms_norm = row * norm_factor * g
        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        output_ptrs = tl.multiple_of(output_ptrs, (16,))
        tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)
        v = rms_norm.to(output_ptr.type.element_ty).to(tl.float32)
        grouped = tl.reshape(v, (BLOCK_SIZE // 128, 128))
        scale = tl.maximum(tl.div_rn(tl.max(tl.abs(grouped), 1), 127.0), 1e-30)
        quant = lib.nearbyint(tl.div_rn(grouped, scale[:, None])).to(tl.int8)
        cols = tl.reshape(col_offsets, (BLOCK_SIZE // 128, 128))
        tl.store(Q + row_idx * n_cols + cols, quant)
        tl.store(S + row_idx * (n_cols // 128) + tl.arange(0, BLOCK_SIZE // 128), scale)
