// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the int8-vllm project
//
// Paged int8 decode attention for gfx908 (CDNA1), int8 per-token-head KV.
//
// Transposed (octave) orientation, so the QK accumulator is already the PV
// B operand:
//   QK: A = K tile [16 tokens x 128], B = Q [128 x NQ queries]
//       C = S^T[token][query]: lane l holds query l%16, tokens 4*(l/16)..+3
//   PV: A = V^T [128 dims x 16 tokens], B = P^T [16 tokens x NQ queries]
//       C = O^T[dim][query]: lane l holds query l%16, dims 4*(l/16)..+3
//
// Softmax over tokens = 4 local adds + xor 16/32 (the four lane groups hold
// the four token quads). P is requantized to int8 in place; V supplies a
// per-token scale that is folded into P before quantization, so the PV MMA
// stays int8 and the i32 accumulator is scaled per KV tile.
//
// One CTA per (request, segment, kv_head). 64-token KV tile staged in LDS:
// K token-major (row stride 144 bytes: 128 + 16 pad, 2-way bank spread),
// V transposed (row stride 68 bytes: 64 + 4 pad, conflict-free A fragments).

#include <cstdint>
#include <cfloat>

#include <hip/hip_bfloat16.h>

#include <cstdlib>

static int getenv_int(const char* k, int d) {
  const char* v = getenv(k);
  return v ? atoi(v) : d;
}

#include "int8_mfma_lane.cuh"

using namespace int8_mfma;

namespace {

constexpr int HDIM = 128;
constexpr int TILE_TOK = 64;      // KV tile in LDS
constexpr int KSTRIDE = 144;      // K row stride (bytes)
constexpr int VSTRIDE = 68;       // V^T row stride (bytes)
constexpr int SMEM_BYTES = TILE_TOK * KSTRIDE + HDIM * VSTRIDE;
constexpr float NEG_BIG = -1e30f;

__device__ __forceinline__ int qk_a_frag(const signed char* k_lds, int token,
                                         int kstep, int quad) {
  return lds_i32(k_lds, token * KSTRIDE + kstep * 16 + quad * 4);
}

__device__ __forceinline__ int pv_a_frag(const signed char* v_lds, int dim,
                                         int token4) {
  return lds_i32(v_lds, dim * VSTRIDE + token4);
}

__device__ __forceinline__ float block_xor16_32(float x) {
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 16);
  x += __shfl_xor_sync(0xffffffffffffffffULL, x, 32);
  return x;
}

// Round-to-nearest non-negative float to i8, clamped.
__device__ __forceinline__ int q_i8(float x) {
  int v = (int)(x + 0.5f);
  return v < 0 ? 0 : (v > 127 ? 127 : v);
}

template <int QG, int QT, int NT>
__global__ void __launch_bounds__(256)
paged_attn_decode_int8_kernel(
    const signed char* __restrict__ q8, const float* __restrict__ q_scale,
    const signed char* __restrict__ k_cache,
    const signed char* __restrict__ v_cache,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const int* __restrict__ block_table, const int* __restrict__ seqused_k,
    const int* __restrict__ cu_q,
    float* __restrict__ seg_out, float* __restrict__ seg_m,
    float* __restrict__ seg_l,
    int num_q_heads, int block_size, int max_blocks,
    int64_t kb_stride, int64_t ks_stride, int64_t kv_stride,
    int64_t vb_stride, int64_t vs_stride, int64_t vv_stride,
    int64_t ksc_b, int64_t ksc_s, int64_t ksc_h,
    int64_t vsc_b, int64_t vsc_s, int64_t vsc_h,
    float qk_scale) {
  extern __shared__ __attribute__((aligned(16))) char smem[];
  signed char* k_lds = reinterpret_cast<signed char*>(smem);
  signed char* v_lds = k_lds + TILE_TOK * KSTRIDE;

  const int seq = blockIdx.x;
  const int seg = blockIdx.y;
  const int kvh = blockIdx.z;
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  constexpr int QW = 16 * NT;            // queries per wave
  constexpr int WAVES = (QT * QG) / QW;  // waves per CTA

  const int q_len = cu_q[seq + 1] - cu_q[seq];
  const int seq_len = seqused_k[seq];
  const int ctx_len = seq_len - q_len;
  if (q_len <= 0 || seq_len <= 0) return;

  const int row0 = cu_q[seq];

  // Segment tile range.
  const int num_tiles = (seq_len + TILE_TOK - 1) / TILE_TOK;
  const int tiles_per_seg = (num_tiles + gridDim.y - 1) / gridDim.y;
  const int tile_lo = seg * tiles_per_seg;
  const int tile_hi = min(tile_lo + tiles_per_seg, num_tiles);
  if (tile_lo >= tile_hi) return;

  // ---- per-lane query columns -------------------------------------------
  const int col = lane & 15;             // query within an N tile
  const int quad = lane >> 4;            // token quad / lds group
  int qi[NT], qrow[NT], qhead[NT];
  float qsc[NT];
  int qfrag[NT][8];
  #pragma unroll
  for (int n = 0; n < NT; ++n) {
    const int qq = wave * QW + n * 16 + col;
    const int i = qq / QG;
    const int h = qq % QG;
    qi[n] = i;
    qrow[n] = row0 + i;
    qhead[n] = kvh * QG + h;
    const bool live = i < q_len;
    qsc[n] = live ? q_scale[qrow[n] * num_q_heads + qhead[n]] : 1.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
      qfrag[n][k] = live
          ? *reinterpret_cast<const int*>(
                q8 + ((int64_t)qrow[n] * num_q_heads + qhead[n]) * HDIM
                     + k * 16 + quad * 4)
          : 0;
    }
  }

  float m_run[NT], l_run[NT];
  float acc[8][NT][4];
  #pragma unroll
  for (int n = 0; n < NT; ++n) {
    m_run[n] = NEG_BIG;
    l_run[n] = 0.0f;
  }
  #pragma unroll
  for (int d = 0; d < 8; ++d)
    #pragma unroll
    for (int n = 0; n < NT; ++n)
      #pragma unroll
      for (int v = 0; v < 4; ++v) acc[d][n][v] = 0.0f;

  for (int tile = tile_lo; tile < tile_hi; ++tile) {
    const int tok0 = tile * TILE_TOK;
    // A 64-token tile never crosses a cache block when block_size % 64 == 0
    // (asserted by the caller), so one lookup serves the whole tile.
    const int blk0 = block_table[seq * max_blocks + tok0 / block_size];
    const int slot0 = tok0 % block_size;

    // ---- stage K (token-major) and V (transposed) ------------------------
    constexpr int THREADS = WAVES * 64;
    for (int idx = threadIdx.x; idx < TILE_TOK * 8; idx += THREADS) {
      const int tok = idx / 8;
      const int ch = idx % 8;                    // 16-byte dim chunk
      const int gtok = tok0 + tok;
      int4 kv = make_int4(0, 0, 0, 0);
      int4 vv = make_int4(0, 0, 0, 0);
      if (gtok < seq_len) {
        const int slot = slot0 + tok;
        const int64_t off = (int64_t)blk0 * kb_stride + slot * ks_stride
                            + kvh * kv_stride + ch * 16;
        kv = *reinterpret_cast<const int4*>(k_cache + off);
        const int64_t voff = (int64_t)blk0 * vb_stride + slot * vs_stride
                             + kvh * vv_stride + ch * 16;
        vv = *reinterpret_cast<const int4*>(v_cache + voff);
      }
      *reinterpret_cast<int4*>(k_lds + tok * KSTRIDE + ch * 16) = kv;
      const signed char* vb = reinterpret_cast<const signed char*>(&vv);
      #pragma unroll
      for (int b = 0; b < 16; ++b)
        v_lds[(ch * 16 + b) * VSTRIDE + tok] = vb[b];
    }
    __syncthreads();

    // ---- QK: S^T[64 tokens][QW queries] ---------------------------------
    float S[4][NT][4];
    #pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      int c0[NT][4];
      #pragma unroll
      for (int n = 0; n < NT; ++n)
        #pragma unroll
        for (int v = 0; v < 4; ++v) c0[n][v] = 0;
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int a = qk_a_frag(k_lds, sub * 16 + col, k, quad);
        #pragma unroll
        for (int n = 0; n < NT; ++n)
          mfma_i8_16x16x16(a, qfrag[n][k], c0[n]);
      }
      const int slot = slot0 + sub * 16 + quad * 4;
      const float* ksp = k_scale + (int64_t)blk0 * ksc_b
                         + slot * ksc_s + kvh * ksc_h;
      float ks[4];
      #pragma unroll
      for (int v = 0; v < 4; ++v) ks[v] = ksp[v * ksc_s];
      #pragma unroll
      for (int n = 0; n < NT; ++n)
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
          const int j = tok0 + sub * 16 + quad * 4 + v;
          const bool ok = (j < seq_len) && (j <= ctx_len + qi[n]);
          S[sub][n][v] = ok
              ? qk_scale * (float)c0[n][v] * qsc[n] * ks[v]
              : NEG_BIG;
        }
    }

    // ---- row max over the 64-token tile ---------------------------------
    float mt[NT];
    #pragma unroll
    for (int n = 0; n < NT; ++n) {
      float mx = NEG_BIG;
      #pragma unroll
      for (int sub = 0; sub < 4; ++sub)
        #pragma unroll
        for (int v = 0; v < 4; ++v) mx = fmaxf(mx, S[sub][n][v]);
      mt[n] = block_xor16_32(mx);
    }
    float rescale[NT], m_new[NT];
    bool dirty = false;
    #pragma unroll
    for (int n = 0; n < NT; ++n) {
      m_new[n] = fmaxf(m_run[n], mt[n]);
      if (qi[n] >= q_len) m_new[n] = 0.0f;  // dead row: keep p at zero
      rescale[n] = exp2f(m_run[n] - m_new[n]);
      m_run[n] = m_new[n];
      dirty |= rescale[n] != 1.0f;
    }
    if (dirty) {
      #pragma unroll
      for (int d = 0; d < 8; ++d)
        #pragma unroll
        for (int n = 0; n < NT; ++n)
          #pragma unroll
          for (int v = 0; v < 4; ++v) acc[d][n][v] *= rescale[n];
      #pragma unroll
      for (int n = 0; n < NT; ++n) l_run[n] *= rescale[n];
    }

    // ---- exp2, row sum, V-scale fold, int8 P ----------------------------
    float p8_scale[NT];   // mp / 127, the i32 accumulator scale
    int p8[4][NT];
    #pragma unroll
    for (int n = 0; n < NT; ++n) {
      float sum = 0.0f, mx = 0.0f;
      float pv[4][4];
      #pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        const int slot = slot0 + sub * 16 + quad * 4;
        const float* vsp = v_scale + (int64_t)blk0 * vsc_b
                           + slot * vsc_s + kvh * vsc_h;
        float vs[4];
        #pragma unroll
        for (int v = 0; v < 4; ++v) vs[v] = vsp[v * vsc_s];
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
          const float p = exp2f(S[sub][n][v] - m_new[n]);
          sum += p;
          const float q = p * vs[v];
          mx = fmaxf(mx, q);
          pv[sub][v] = q;
        }
      }
      sum = block_xor16_32(sum);
      mx = block_xor16_32(mx);
      l_run[n] += sum;
      const float inv = mx > 0.0f ? 127.0f / mx : 0.0f;
      p8_scale[n] = mx * (1.0f / 127.0f);
      #pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        int packed = 0;
        #pragma unroll
        for (int v = 0; v < 4; ++v)
          packed |= q_i8(pv[sub][v] * inv) << (8 * v);
        p8[sub][n] = packed;
      }
    }

    // ---- PV: accumulate int8 products, one i32 tile set at a time -------
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
      const int dim = d * 16 + col;
      #pragma unroll
      for (int n = 0; n < NT; ++n) {
        int c[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
          const int a = pv_a_frag(v_lds, dim, sub * 16 + quad * 4);
          mfma_i8_16x16x16(a, p8[sub][n], c);
        }
        const float s = p8_scale[n];
        #pragma unroll
        for (int v = 0; v < 4; ++v) acc[d][n][v] += (float)c[v] * s;
      }
    }
    __syncthreads();
  }

  // ---- write segment partials -------------------------------------------
  #pragma unroll
  for (int n = 0; n < NT; ++n) {
    if (qi[n] >= q_len) continue;
    const int64_t base = ((int64_t)qrow[n] * num_q_heads + qhead[n])
                         * gridDim.y + seg;
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
      #pragma unroll
      for (int v = 0; v < 4; ++v)
        seg_out[(base * HDIM) + d * 16 + quad * 4 + v] = acc[d][n][v];
    }
    if (quad == 0) {
      seg_m[base] = m_run[n];
      seg_l[base] = l_run[n];
    }
  }
}

// ---- Q pack: bf16 [rows, heads, 128] -> int8 + per-row scale -------------

__global__ void pack_q_kernel(const hip_bfloat16* __restrict__ q,
                             signed char* __restrict__ q8,
                             float* __restrict__ qsc, int64_t rows_total) {
  __shared__ float red[128];
  const int64_t row = blockIdx.x;
  const int d = threadIdx.x;             // one head-dim element per thread
  const float x = (float)q[row * HDIM + d];
  red[d] = fabsf(x);
  __syncthreads();
  for (int s = 64; s > 0; s >>= 1) {
    if (d < s) red[d] = fmaxf(red[d], red[d + s]);
    __syncthreads();
  }
  const float amax = red[0];
  const float scale = amax > 0.0f ? amax * (1.0f / 127.0f) : 1.0f;
  if (d == 0) qsc[row] = scale;
  q8[row * HDIM + d] = (signed char)(x / scale);
}

// ---- segment reduce ------------------------------------------------------

__global__ void reduce_segments_kernel(
    const float* __restrict__ seg_out, const float* __restrict__ seg_m,
    const float* __restrict__ seg_l, hip_bfloat16* __restrict__ out,
    int64_t rows_total, int num_seqs, int num_q_heads, int num_segments) {
  const int64_t row = blockIdx.x;
  const int d = threadIdx.x;
  if (row >= rows_total) return;
  float mx = -FLT_MAX;
  for (int s = 0; s < num_segments; ++s) {
    const int64_t b = row * num_segments + s;
    mx = fmaxf(mx, seg_m[b]);
  }
  float num = 0.0f, den = 0.0f;
  for (int s = 0; s < num_segments; ++s) {
    const int64_t b = row * num_segments + s;
    const float w = exp2f(seg_m[b] - mx);
    num += seg_out[b * HDIM + d] * w;
    den += seg_l[b] * w;
  }
  out[row * HDIM + d] = (hip_bfloat16)(num / den);
}

}  // namespace

extern "C" int launch_pack_q(const void* q, void* q8, void* qsc,
                             int64_t rows_total, void* stream) {
  pack_q_kernel<<<(int)rows_total, 128, 0,
                  reinterpret_cast<hipStream_t>(stream)>>>(
      reinterpret_cast<const hip_bfloat16*>(q),
      reinterpret_cast<signed char*>(q8), reinterpret_cast<float*>(qsc),
      rows_total);
  return (int)hipGetLastError();
}

extern "C" int launch_paged_attn_decode_int8(
    const void* q8, const void* q_scale, const void* k_cache,
    const void* v_cache, const void* k_scale, const void* v_scale,
    const void* block_table, const void* seqused_k, const void* cu_q,
    void* seg_out, void* seg_m, void* seg_l,
    int num_seqs, int num_segments, int num_kv_heads, int num_q_heads,
    int qg, int block_size, int max_blocks,
    int64_t kb_stride, int64_t ks_stride, int64_t kv_stride,
    int64_t vb_stride, int64_t vs_stride, int64_t vv_stride,
    int64_t ksc_b, int64_t ksc_s, int64_t ksc_h,
    int64_t vsc_b, int64_t vsc_s, int64_t vsc_h,
    float qk_scale, void* stream) {
  dim3 grid(num_seqs, num_segments, num_kv_heads);
  const int smem = SMEM_BYTES;
  const int nt = qg == 8 ? (int)getenv_int("INT8_ATTN_NT", 2) : 1;
  if (qg == 8 && nt == 1) {
    constexpr int waves = (8 * 8) / 16;
    paged_attn_decode_int8_kernel<8, 8, 1><<<grid, waves * 64, smem,
        reinterpret_cast<hipStream_t>(stream)>>>(
        (const signed char*)q8, (const float*)q_scale,
        (const signed char*)k_cache, (const signed char*)v_cache,
        (const float*)k_scale, (const float*)v_scale,
        (const int*)block_table, (const int*)seqused_k, (const int*)cu_q,
        (float*)seg_out, (float*)seg_m, (float*)seg_l, num_q_heads,
        block_size, max_blocks, kb_stride, ks_stride, kv_stride, vb_stride,
        vs_stride, vv_stride, ksc_b, ksc_s, ksc_h, vsc_b, vsc_s, vsc_h,
        qk_scale);
  } else if (qg == 8) {
    constexpr int waves = (8 * 8) / 32;
    paged_attn_decode_int8_kernel<8, 8, 2><<<grid, waves * 64, smem,
        reinterpret_cast<hipStream_t>(stream)>>>(
        (const signed char*)q8, (const float*)q_scale,
        (const signed char*)k_cache, (const signed char*)v_cache,
        (const float*)k_scale, (const float*)v_scale,
        (const int*)block_table, (const int*)seqused_k, (const int*)cu_q,
        (float*)seg_out, (float*)seg_m, (float*)seg_l, num_q_heads,
        block_size, max_blocks, kb_stride, ks_stride, kv_stride, vb_stride,
        vs_stride, vv_stride, ksc_b, ksc_s, ksc_h, vsc_b, vsc_s, vsc_h,
        qk_scale);
  } else if (qg == 4) {
    constexpr int waves = (8 * 4) / 16;
    paged_attn_decode_int8_kernel<4, 8, 1><<<grid, waves * 64, smem,
        reinterpret_cast<hipStream_t>(stream)>>>(
        (const signed char*)q8, (const float*)q_scale,
        (const signed char*)k_cache, (const signed char*)v_cache,
        (const float*)k_scale, (const float*)v_scale,
        (const int*)block_table, (const int*)seqused_k, (const int*)cu_q,
        (float*)seg_out, (float*)seg_m, (float*)seg_l, num_q_heads,
        block_size, max_blocks, kb_stride, ks_stride, kv_stride, vb_stride,
        vs_stride, vv_stride, ksc_b, ksc_s, ksc_h, vsc_b, vsc_s, vsc_h,
        qk_scale);
  } else {
    return -2;
  }
  return (int)hipGetLastError();
}

extern "C" int launch_reduce_segments(
    const void* seg_out, const void* seg_m, const void* seg_l, void* out,
    int64_t rows_total, int num_seqs, int num_q_heads, int num_segments,
    void* stream) {
  reduce_segments_kernel<<<(int)rows_total, 128, 0,
                           reinterpret_cast<hipStream_t>(stream)>>>(
      (const float*)seg_out, (const float*)seg_m, (const float*)seg_l,
      (hip_bfloat16*)out, rows_total, num_seqs, num_q_heads,
      num_segments);
  return (int)hipGetLastError();
}
