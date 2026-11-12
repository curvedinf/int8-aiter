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
// One wave (64 lanes) computes a BM x BN tile (BN/16 16x16 B/C fragments
// share the lane's A fragment). The 128-K group loop applies the group scale
// to the int32 dot, so KSPLIT partials are plain fp32 sums: with KSPLIT>1 the
// launch zeroes a fp32 buffer, each part atomically adds its partial, and a
// convert kernel writes bf16. KSPLIT keeps the weight traffic at ceil(M/BM)
// passes while filling the 120 CUs (the Triton configs use 4 here).

#include <cstdint>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "int8_mfma_lane.cuh"

using int8_mfma::mfma_i8_16x16x16;

namespace {

template <int BM, int BN, int KSPLIT>
__global__ void __launch_bounds__(64) gemm_a8w8_m16_kernel(
    const int* __restrict__ a32,      // A as int32 [M, K/4]
    const int* __restrict__ wp,       // [N/16, K/16, 64] int32
    const float* __restrict__ a_s,    // [M, K/128]
    const float* __restrict__ w_s,    // [N, K/128]
    hip_bfloat16* __restrict__ out,   // [M, N] when KSPLIT==1
    float* __restrict__ partial,      // [M, N] when KSPLIT>1
    int M, int N, int K) {
  constexpr int MT = BM / 16;
  constexpr int NT = BN / 16;
  const int lane = threadIdx.x;
  const int col = lane & 15;
  const int quad = lane >> 4;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int k16n = K / 16;
  const int groups = K / 128;
  const int part = KSPLIT > 1 ? (int)blockIdx.z : 0;
  const int g0 = part * (groups / KSPLIT);
  const int g1 = (part + 1) * (groups / KSPLIT);

  float outc[MT][NT][4];
  #pragma unroll
  for (int mi = 0; mi < MT; ++mi)
    #pragma unroll
    for (int j = 0; j < NT; ++j)
      #pragma unroll
      for (int i = 0; i < 4; ++i) outc[mi][j][i] = 0.0f;

  for (int g = g0; g < g1; ++g) {
    int acc[MT][NT][4];
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi)
      #pragma unroll
      for (int j = 0; j < NT; ++j)
        #pragma unroll
        for (int i = 0; i < 4; ++i) acc[mi][j][i] = 0;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      int aop[4][MT];
      int bop[4][NT];
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
        for (int j = 0; j < NT; ++j) {
          const int nt = blockIdx.x * NT + j;
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
          for (int j = 0; j < NT; ++j) {
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
    for (int j = 0; j < NT; ++j) {
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
  for (int j = 0; j < NT; ++j) {
    const int n = n0 + j * 16 + col;
    if (n >= N) continue;
    #pragma unroll
    for (int mi = 0; mi < MT; ++mi)
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int m = m0 + mi * 16 + 4 * quad + i;
        if (m >= M) continue;
        if (KSPLIT == 1) {
          out[(size_t)m * N + n] = hip_bfloat16(outc[mi][j][i]);
        } else {
          atomicAdd(&partial[(size_t)m * N + n], outc[mi][j][i]);
        }
      }
  }
}

__global__ void convert_kernel(const float* __restrict__ partial,
                               hip_bfloat16* __restrict__ out,
                               size_t count) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = hip_bfloat16(partial[i]);
}

}  // namespace

extern "C" int launch_gemm_a8w8_m16(
    const void* a32, const void* wp, const float* a_s, const float* w_s,
    void* out, void* partial, int M, int N, int K, int block_m, int block_n,
    int ksplit, void* stream) {
  // BM 32/64 spill registers today; keep auto at the measured-best 16x32.
  const int bm = block_m ? block_m : 16;
  const int bn = block_n ? block_n : 32;
  const int ks = ksplit ? ksplit : 1;
  if (ks > 1 && partial == nullptr) return -3;
  const dim3 grid((N + bn - 1) / bn, (M + bm - 1) / bm, ks);
  auto stream_ = reinterpret_cast<hipStream_t>(stream);
  auto out_ = (hip_bfloat16*)out;
  auto part_ = (float*)partial;
  if (ks > 1) {
    hipMemsetAsync(part_, 0, (size_t)M * N * sizeof(float), stream_);
  }
  #define LAUNCH(BM_, BN_, KS_)                                                \
    gemm_a8w8_m16_kernel<BM_, BN_, KS_><<<grid, 64, 0, stream_>>>(             \
        (const int*)a32, (const int*)wp, a_s, w_s, out_, part_, M, N, K)
  if (bm == 16 && bn == 64 && ks == 1) LAUNCH(16, 64, 1);
  else if (bm == 16 && bn == 32 && ks == 1) LAUNCH(16, 32, 1);
  else if (bm == 32 && bn == 32 && ks == 1) LAUNCH(32, 32, 1);
  else if (bm == 32 && bn == 64 && ks == 1) LAUNCH(32, 64, 1);
  else if (bm == 32 && bn == 64 && ks == 2) LAUNCH(32, 64, 2);
  else if (bm == 32 && bn == 64 && ks == 4) LAUNCH(32, 64, 4);
  else if (bm == 32 && bn == 128 && ks == 4) LAUNCH(32, 128, 4);
  else if (bm == 16 && bn == 64 && ks == 4) LAUNCH(16, 64, 4);
  else return -2;
  #undef LAUNCH
  if (ks > 1) {
    const int n = M * N;
    convert_kernel<<<(n + 255) / 256, 256, 0, stream_>>>(part_, out_,
                                                         (size_t)n);
  }
  return (int)hipGetLastError();
}
