// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the int8-vllm project
//
// gfx908 (CDNA1) int8 MFMA fragment primitives.
//
// One instruction matters here: v_mfma_i32_16x16x16i8. Its register layout,
// confirmed by the bit-exact self-test in int8_mfma_selftest.hip:
//
//   A (16x16 int8)  lane l holds row l%16, k = 4*(l/16) .. +3   (1 i32)
//   B (16x16 int8)  lane l holds col l%16, k = 4*(l/16) .. +3   (1 i32)
//   C (16x16 i32)   lane l holds col l%16, row = 4*(l/16) .. +3 (4 i32)
//
// The asymmetry is the useful part: C's four values are four *tokens* for one
// query column, so a softmax over tokens is four local adds plus an xor
// reduce over lane groups (bits 4 and 5), and the same four values are
// already the B fragment of the PV MMA — no LDS round trip for P.
//
// HIP compiles a kernel body for host as well; the real implementations are
// device-only and the host pass gets inert stubs so the launch builds.

#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace int8_mfma {

constexpr int WAVE = 64;
constexpr int TILE = 16;

// ---- lane maps -------------------------------------------------------------

__host__ __device__ __forceinline__ int lane_col(int lane) { return lane & 15; }
__host__ __device__ __forceinline__ int lane_quad(int lane) { return lane >> 4; }

// ---- i8 x4 packing ---------------------------------------------------------

__host__ __device__ __forceinline__ int pack_i8x4(const signed char* p) {
  return (int)(unsigned)(unsigned char)p[0] |
         (int)((unsigned)(unsigned char)p[1] << 8) |
         (int)((unsigned)(unsigned char)p[2] << 16) |
         (int)((unsigned)(unsigned char)p[3] << 24);
}

__host__ __device__ __forceinline__ void unpack_i8x4(int v, signed char* p) {
  p[0] = (signed char)(v & 0xff);
  p[1] = (signed char)((v >> 8) & 0xff);
  p[2] = (signed char)((v >> 16) & 0xff);
  p[3] = (signed char)((v >> 24) & 0xff);
}

#if defined(__HIP_DEVICE_COMPILE__)

// ---- MMA -------------------------------------------------------------------

typedef int v4i32 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void mfma_i8_16x16x16(int a, int b, int c[4]) {
  v4i32 cv = {c[0], c[1], c[2], c[3]};
  cv = __builtin_amdgcn_mfma_i32_16x16x16i8(a, b, cv, 0, 0, 0);
  c[0] = cv[0];
  c[1] = cv[1];
  c[2] = cv[2];
  c[3] = cv[3];
}

// 16 contiguous int8 of one row from LDS: one ds_read_b128. For the 16-wide
// k block `k16`, the four int fields are the four lane groups' A operands;
// a lane uses field `quad` (see k16_lane_operand).
// `row_stride` is in bytes.
__device__ __forceinline__ int4 lds_row16(
    const void* base, int row, int row_stride, int k16) {
  const int4* p = reinterpret_cast<const int4*>(
      reinterpret_cast<const char*>(base) + row * row_stride + k16 * 16);
  return *p;
}

__device__ __forceinline__ int lds_i32(const void* base, int byte_off) {
  return *reinterpret_cast<const int*>(
      reinterpret_cast<const char*>(base) + byte_off);
}

// Fields of a 16-byte A load; this lane's operand is field `quad`, holding
// k = 16*k16 + 4*quad .. +3.
__device__ __forceinline__ void split_k16(int4 v, int a[4]) {
  a[0] = v.x;
  a[1] = v.y;
  a[2] = v.z;
  a[3] = v.w;
}

__device__ __forceinline__ int k16_lane_operand(int4 v, int quad) {
  return quad == 0 ? v.x : quad == 1 ? v.y : quad == 2 ? v.z : v.w;
}

// ---- C fragment helpers ----------------------------------------------------

__device__ __forceinline__ void c_store_row4(
    int* out, int n, int col, int quad, const int c[4]) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) out[(4 * quad + i) * n + col] = c[i];
}

__device__ __forceinline__ void c_scale(const int c[4], float s, float out[4]) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = (float)c[i] * s;
}

// Sum of this lane's four tokens for its query column, before the cross-group
// reduce. Softmax over tokens = this + xor_reduce_groups.
__device__ __forceinline__ float c_token_sum(const float c[4]) {
  return (c[0] + c[1]) + (c[2] + c[3]);
}

// Reduce across the four lane groups that share a query column.
__device__ __forceinline__ float xor_reduce_groups(float x) {
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 16);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 32);
  return x;
}

#else  // host pass: inert stubs so the launch and kernel body compile

__host__ __device__ inline void mfma_i8_16x16x16(int, int, int[4]) {}
__host__ __device__ inline int4 lds_row16(const void*, int, int, int) { return int4(); }
__host__ __device__ inline int lds_i32(const void*, int) { return 0; }
__host__ __device__ inline void split_k16(int4, int a[4]) { a[0] = a[1] = a[2] = a[3] = 0; }
__host__ __device__ inline int k16_lane_operand(int4, int) { return 0; }
__host__ __device__ inline void c_store_row4(int*, int, int, int, const int[4]) {}
__host__ __device__ inline void c_scale(const int[4], float, float[4]) {}
__host__ __device__ inline float c_token_sum(const float[4]) { return 0.0f; }
__host__ __device__ inline float xor_reduce_groups(float) { return 0.0f; }

#endif  // __HIP_DEVICE_COMPILE__

}  // namespace int8_mfma
