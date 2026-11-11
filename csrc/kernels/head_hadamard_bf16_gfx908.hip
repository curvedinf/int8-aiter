// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the int8-vllm project
//
// Normalized 64-point Walsh-Hadamard rotation over the head dimension.
//
// out[..., 64-block] = H64 @ x / 8, with H64 @ H64 = 64*I, so the transform is
// orthonormal and self-inverse: applying it twice returns the input. Used on
// Q/K/V before int8 KV quantization and on the attention output to undo V's
// rotation (llama.cpp #21038). One wave per 64-value block; the butterfly is
// register-only (six shfl_xor stages).

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

namespace {

constexpr int BLOCK = 64;

__global__ void __launch_bounds__(BLOCK) head_hadamard_kernel(
    const hip_bfloat16* __restrict__ in, hip_bfloat16* __restrict__ out,
    int blocks_per_row, int dim) {
  const int gid = blockIdx.x;
  const int blk = gid % blocks_per_row;
  const int row = gid / blocks_per_row;
  const int lane = threadIdx.x;
  const int off = row * dim + blk * BLOCK + lane;

  float x = (float)in[off];
  #pragma unroll
  for (int s = 1; s < BLOCK; s <<= 1) {
    const float y = __shfl_xor_sync(0xffffffffffffffffULL, x, s);
    x = (lane & s) ? (y - x) : (x + y);
  }
  out[off] = (hip_bfloat16)(x * 0.125f);
}

}  // namespace

extern "C" int launch_head_hadamard(const void* in, void* out, int64_t rows,
                                    int dim, void* stream) {
  if (dim % BLOCK != 0) return -2;
  const int blocks_per_row = dim / BLOCK;
  const int64_t programs = rows * blocks_per_row;
  head_hadamard_kernel<<<(int)programs, BLOCK, 0,
                         reinterpret_cast<hipStream_t>(stream)>>>(
      reinterpret_cast<const hip_bfloat16*>(in),
      reinterpret_cast<hip_bfloat16*>(out), blocks_per_row, dim);
  return (int)hipGetLastError();
}
