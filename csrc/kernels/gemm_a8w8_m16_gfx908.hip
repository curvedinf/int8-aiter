// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the int8-vllm project
//
// Small-M int8 A8W8 GEMM for gfx908 (CDNA1), GS128 group scales.
//
//   out[m,n] = sum_g a_s[m,g] * w_s[n,g] * sum_{k in group g} A[m,k] * W[n,k]
//
// A: int8 [M,K] row-major. W: int8 [N,K] row-major, PRESHUFFLED at load into
// the B-fragment order wp[n_tile][k16][lane] = W32[n_tile*16 + lane%16,
// k16*4 + lane/16] (int32 = four k bytes). a_s: f32 [M,K/128].
// w_s: f32 [N,K/128]. out: bf16 [M,N] row-major.
//
// One wave (64 lanes) computes a BLOCK_M x 64 tile: four 16x16 B/C fragments
// share the lane's A fragment, so the wave issues four MFMAs per 16-K step.
// A is staged per 16-K step through 256 B of LDS in operand order; the group
// scale is applied every 128 K so the int32 dot never leaves the accumulator
// unscaled across groups. M/N guards cover partial tiles.

#include <cstdint>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "int8_mfma_lane.cuh"

using int8_mfma::mfma_i8_16x16x16;

namespace {

constexpr int kBlockM = 16;   // rows per block
constexpr int kBlockN = 64;   // columns per block (4 fragments)

template <int BM>  // BM = 16, 32 or 64 rows per CTA
__global__ void __launch_bounds__(64) gemm_a8w8_m16_kernel(
    const int* __restrict__ a32,      // A as int32 [M, K/4]
    const int* __restrict__ wp,       // [N/16, K/16, 64] int32
    const float* __restrict__ a_s,    // [M, K/128]
    const float* __restrict__ w_s,    // [N, K/128]
    hip_bfloat16* __restrict__ out,   // [M, N]
    int M, int N, int K) {
  constexpr int MT = BM / 16;         // 16-row M tiles per CTA
  const int lane = threadIdx.x;
  const int col = lane & 15;
  const int quad = lane >> 4;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * kBlockN;
  const int k16n = K / 16;
  const int groups = K / 128;

  float outc[MT][kBlockN / 16][4];
  #pragma unroll
  for (int mi = 0; mi < MT; ++mi)
    #pragma unroll
    for (int j = 0; j < kBlockN / 16; ++j)
      #pragma unroll
      for (int i = 0; i < 4; ++i) outc[mi][j][i] = 0.0f;

  for (int g = 0; g < groups; ++g) {
    int acc[MT][kBlockN / 16][4];
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi)
      #pragma unroll
      for (int j = 0; j < kBlockN / 16; ++j)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[mi][j][i] = 0;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      int aop[4][MT];
      int bop[4][kBlockN / 16];
      #pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int k16 = g * 8 + h * 4 + s;
        #pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
          const int row = m0 + mi * 16 + col;
          aop[s][mi] = (row < M)
              ? a32[(size_t)row * (K / 4) + k16 * 4 + quad] : 0;
        }
        #pragma unroll
        for (int j = 0; j < kBlockN / 16; ++j) {
          const int nt = blockIdx.x * (kBlockN / 16) + j;
          const int n = n0 + j * 16 + col;
          bop[s][j] = (n < N)
              ? wp[(((size_t)nt * k16n) + k16) * 64 + lane] : 0;
        }
      }
      #pragma unroll
      for (int s = 0; s < 4; ++s) {
        #pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
          #pragma unroll
          for (int j = 0; j < kBlockN / 16; ++j) {
            mfma_i8_16x16x16(aop[s][mi], bop[s][j], acc[mi][j]);
          }
        }
      }
    }
    float as[MT][4];
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi)
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int m = m0 + mi * 16 + 4 * quad + i;
        as[mi][i] = (m < M) ? a_s[(size_t)m * groups + g] : 0.0f;
      }
    #pragma unroll
    for (int j = 0; j < kBlockN / 16; ++j) {
      const int n = n0 + j * 16 + col;
      const float ws = (n < N) ? w_s[(size_t)n * groups + g] : 0.0f;
      #pragma unroll
      for (int mi = 0; mi < MT; ++mi)
        #pragma unroll
        for (int i = 0; i < 4; ++i)
          outc[mi][j][i] += (float)acc[mi][j][i] * as[mi][i] * ws;
    }
  }

  #pragma unroll
  for (int j = 0; j < kBlockN / 16; ++j) {
    const int n = n0 + j * 16 + col;
    if (n >= N) continue;
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi)
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int m = m0 + mi * 16 + 4 * quad + i;
        if (m < M) out[(size_t)m * N + n] = hip_bfloat16(outc[mi][j][i]);
      }
  }
}

}  // namespace

extern "C" int launch_gemm_a8w8_m16(
    const void* a32, const void* wp, const float* a_s, const float* w_s,
    void* out, int M, int N, int K, int block_m, void* stream) {
  // BM 32/64 spill registers today; keep auto at the measured-best 16.
  const int bm = block_m ? block_m : 16;
  const dim3 grid((N + kBlockN - 1) / kBlockN, (M + bm - 1) / bm);
  auto stream_ = reinterpret_cast<hipStream_t>(stream);
  if (bm == 16) {
    gemm_a8w8_m16_kernel<16><<<grid, 64, 0, stream_>>>(
        (const int*)a32, (const int*)wp, a_s, w_s, (hip_bfloat16*)out, M, N, K);
  } else if (bm == 32) {
    gemm_a8w8_m16_kernel<32><<<grid, 64, 0, stream_>>>(
        (const int*)a32, (const int*)wp, a_s, w_s, (hip_bfloat16*)out, M, N, K);
  } else if (bm == 64) {
    gemm_a8w8_m16_kernel<64><<<grid, 64, 0, stream_>>>(
        (const int*)a32, (const int*)wp, a_s, w_s, (hip_bfloat16*)out, M, N, K);
  } else {
    return -2;
  }
  return (int)hipGetLastError();
}
