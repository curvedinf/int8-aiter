// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the int8-vllm project
//
// Bit-exact self-test for the gfx908 int8 MFMA primitives.
//
// Three checks, all against a torch reference on the host:
//   1. direct global-memory pack  -> validates the A/B/C lane maps
//   2. LDS-staged A via ds_read_b128 -> validates the 16-byte fragment load
//   3. column-sum reduction       -> validates the softmax-over-tokens pattern
//
// One block, 64 lanes, 16x16x16 tiles.

#include <cstdint>

#include "int8_mfma_lane.cuh"

using namespace int8_mfma;

namespace {

__global__ void selftest_kernel(const signed char* __restrict__ A,
                                const signed char* __restrict__ B,
                                int* __restrict__ C_direct,
                                int* __restrict__ C_lds,
                                float* __restrict__ colsum) {
  const int lane = threadIdx.x;
  const int col = lane_col(lane);
  const int quad = lane_quad(lane);

  // ---- check 1: pack straight from global memory ---------------------------
  int a = pack_i8x4(A + col * TILE + quad * 4);
  signed char bbuf[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) bbuf[i] = B[(quad * 4 + i) * TILE + col];
  int b = pack_i8x4(bbuf);

  int c[4] = {0, 0, 0, 0};
  mfma_i8_16x16x16(a, b, c);
  c_store_row4(C_direct, TILE, col, quad, c);

  // ---- check 3 input: column sum of the same accumulator -------------------
  float cf[4];
  c_scale(c, 1.0f, cf);
  float s = xor_reduce_groups(c_token_sum(cf));
  if (quad == 0) colsum[col] = s;

  // ---- check 2: stage A in LDS, load 16 bytes per lane (4 k-steps) ---------
  __shared__ __attribute__((aligned(16))) signed char smem[TILE * TILE];
  #pragma unroll
  for (int i = 0; i < 4; ++i) smem[col * TILE + quad * 4 + i] = A[col * TILE + quad * 4 + i];
  __syncthreads();

  int4 row = lds_row16(smem, col, TILE, 0);
  int a_lds = k16_lane_operand(row, quad);
  signed char bk[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) bk[i] = B[(4 * quad + i) * TILE + col];

  int cl[4] = {0, 0, 0, 0};
  mfma_i8_16x16x16(a_lds, pack_i8x4(bk), cl);
  c_store_row4(C_lds, TILE, col, quad, cl);
}

}  // namespace

extern "C" int launch_int8_selftest(const void* A, const void* B,
                                    void* C_direct, void* C_lds, void* colsum,
                                    void* stream) {
  selftest_kernel<<<1, WAVE, 0, reinterpret_cast<hipStream_t>(stream)>>>(
      reinterpret_cast<const signed char*>(A),
      reinterpret_cast<const signed char*>(B),
      reinterpret_cast<int*>(C_direct), reinterpret_cast<int*>(C_lds),
      reinterpret_cast<float*>(colsum));
  return (int)hipGetLastError();
}
