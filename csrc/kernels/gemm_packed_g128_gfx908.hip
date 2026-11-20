// SPDX-License-Identifier: Apache-2.0
// Register-local packed-weight to GS128 INT8 conversion and gfx908 MFMA.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <hip/hip_fp16.h>
using I4 = int __attribute__((ext_vector_type(4)));
enum PackedCodec
{
    NVFP4,
    Q8_0,
    IQ2_XXS,
    AFFINE_W4,
    SIGNED_W8,
    WAVE_W4,
    WAVE_W8,
    VECTOR_W4,
    VECTOR_W8,
    OVERLAP_W4
};

template <int Codec>
constexpr bool is_vector = Codec == VECTOR_W4 || Codec == VECTOR_W8;
template <int Codec>
constexpr bool is_w4 = Codec == AFFINE_W4 || Codec == WAVE_W4 || Codec == VECTOR_W4;
template <int Codec>
constexpr bool is_w8 = Codec == SIGNED_W8 || Codec == WAVE_W8 || Codec == VECTOR_W8;
template <int Codec>
constexpr bool is_wave = Codec == WAVE_W4 || Codec == WAVE_W8 || is_vector<Codec>;

template <int Codec>
__device__ __forceinline__ int64_t weight_group_index(int channel, int groups, int group)
{
    if constexpr(is_wave<Codec>)
        return (static_cast<int64_t>(channel / 16) * groups + group) * 16 + channel % 16;
    return static_cast<int64_t>(channel) * groups + group;
}

// One word holds four weights consumed by one lane of an MFMA. Its 64 lanes
// read adjacent words from a prepacked N16 by K16 microtile in wave order.
template <int Codec>
__device__ __forceinline__ int64_t weight_word_index(int channel, int width, int column)
{
    const int64_t output_tile = channel / 16;
    const int channel_lane    = channel % 16;
    const int input_lane      = (column % 16) / 4;
    if constexpr(is_vector<Codec>)
    {
        const int64_t group = output_tile * (width / 128) + column / 128;
        const int64_t lane  = (group * 4 + input_lane) * 16 + channel_lane;
        return lane * 8 + (column % 128) / 16;
    }
    if constexpr(is_wave<Codec>)
    {
        const int64_t tile = output_tile * (width / 16) + column / 16;
        return (tile * 4 + input_lane) * 16 + channel_lane;
    }
    return (static_cast<int64_t>(channel) * width + column) / 4;
}

template <int CODEC>
__device__ __forceinline__ float group_scale(const float* gs, const uint8_t* ws, int64_t index)
{
    if constexpr(is_w4<CODEC> || is_w8<CODEC>)
        return __half2float(reinterpret_cast<const __half*>(ws)[index]);
    else
        return gs[index];
}

template <int CODEC>
__device__ __forceinline__ float affine_dot(int dot,
                                            const int32_t* sums,
                                            int64_t activation_group,
                                            const uint8_t* zeros,
                                            int64_t weight_group)
{
    if constexpr(is_w4<CODEC>)
        return float(dot) - __half2float(reinterpret_cast<const __half*>(zeros)[weight_group]) *
                                float(sums[activation_group]);
    else
        return float(dot);
}
__device__ __forceinline__ float fp4(int v) {
 int m=v&7;
 float x=m<2?m*0.5f:__uint_as_float(((m/2+126)<<23)|((m&1)<<22));
 return (v&8)?-x:x;
}
__device__ __forceinline__ float fp8(int v) {
 int e=(v>>3)&15,m=v&7;
 float x=e?__uint_as_float(((e+120)<<23)|(m<<20)):m*(1.0f/512);
 return (v&128)?-x:x;
}
__device__ __forceinline__ uint32_t expand_four_nibbles(uint16_t nibble)
{
    uint32_t pair = (nibble & 0xffu) | ((uint32_t(nibble) & 0xff00u) << 8);
    return (pair & 0x000f000fu) | ((pair & 0x00f000f0u) << 4);
}
using FourBitGroup  = uint16_t __attribute__((ext_vector_type(8)));
using EightBitGroup = uint32_t __attribute__((ext_vector_type(8)));
template <int Codec>
struct LaneGroup
{
    FourBitGroup w4;
    EightBitGroup w8;
    __device__ __forceinline__ void load(const uint8_t* w, int row, int k, int col)
    {
        const auto word = weight_word_index<Codec>(row, k, col);
        if constexpr(Codec == VECTOR_W4)
            w4 = *reinterpret_cast<const FourBitGroup*>(w + 2 * word);
        if constexpr(Codec == VECTOR_W8)
            w8 = *reinterpret_cast<const EightBitGroup*>(w + 4 * word);
    }
    __device__ __forceinline__ uint32_t get(int sub) const
    {
        if constexpr(Codec == VECTOR_W4)
            return expand_four_nibbles(w4[sub]);
        if constexpr(Codec == VECTOR_W8)
            return w8[sub];
        return 0;
    }
};

template<int CODEC,bool HINT=false>
__device__ __forceinline__ uint32_t decode4(const uint8_t* W,const uint8_t* WS,
 const uint8_t* GRID,int n,int K,uint32_t k,float wg,float inv,float d_hint=0.f){
 uint32_t result=0;
 if constexpr(CODEC==0){
  uint16_t packed=*reinterpret_cast<const uint16_t*>(W+(int64_t)n*(K/2)+k/2);
  float scale=fp8(WS[(int64_t)n*(K/16)+k/16]);
  #pragma unroll
  for(int j=0;j<4;++j){
   float v=(fp4((packed>>(j*4))&15)*scale)*wg;
   result|=(uint32_t)(uint8_t)(int8_t)__float2int_rn(v*inv)<<(j*8);
  }
 }else if constexpr(CODEC==1){
  const uint8_t* base=W+(int64_t)n*(K/32)*34+(k/32)*34;
  float scale=__half2float(*reinterpret_cast<const __half*>(base));
  uint32_t packed;__builtin_memcpy(&packed,base+2+k%32,4);
  #pragma unroll
  for(int j=0;j<4;++j){
   float v=(float)(int8_t)(packed>>(j*8))*scale;
   result|=(uint32_t)(uint8_t)(int8_t)__float2int_rn(v*inv)<<(j*8);
  }
 }
 else if constexpr(is_w4<CODEC>)
 {
     uint16_t nibble;
     __builtin_memcpy(&nibble, W + 2 * weight_word_index<CODEC>(n, K, k), sizeof(nibble));
     result = expand_four_nibbles(nibble);
 }
 else if constexpr(is_w8<CODEC>)
 {
     __builtin_memcpy(&result, W + 4 * weight_word_index<CODEC>(n, K, k), sizeof(result));
 }
 else
 {
     const uint8_t* base = W + (int64_t)n * (K / 256) * 66 + (k / 256) * 66;
     float d             = HINT ? d_hint : __half2float(*reinterpret_cast<const __half*>(base));
     int sub = (k % 256) / 32, lane = (k % 32) / 8, j0 = k % 8;
     int code = base[2 + sub * 8 + lane];
     uint32_t bits;
     __builtin_memcpy(&bits, base + 6 + sub * 8, 4);
     int sign7 = (bits >> (7 * lane)) & 127, signs = sign7 | ((__popc(sign7) & 1) << 7);
     float scale = (d * (0.5f + (float)(bits >> 28))) * 0.25f;
     uint32_t values;
     __builtin_memcpy(&values, GRID + code * 8 + j0, 4);
  #pragma unroll
  for(int j=0;j<4;++j){
   float v=scale*(float)((values>>(j*8))&255);
   if((signs>>(j0+j))&1)v=-v;
   result|=(uint32_t)(uint8_t)(int8_t)__float2int_rn(v*inv)<<(j*8);
  }
 }
 return result;
}
template <int MT, int CODEC>
__global__ void packed_all(const int8_t* X,
                           const float* XS,
                           const uint8_t* W,
                           const uint8_t* WS,
                           const float* WG,
                           const float* GS,
                           const uint8_t* GRID,
                           float* Y,
                           int M,
                           int N,
                           int K,
                           int start,
                           int L,
                           int per_row,
                           int split,
                           const int32_t* sums = nullptr)
{
    int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
    int n              = blockIdx.y * 128 + wave * 16 + lane % 16;
    int m0             = blockIdx.x * (16 * MT);
    float accum[MT][4] = {};
    int ng = (L + 127) / 128, span = (ng + split - 1) / split;
    int begin = blockIdx.z * span, end = min(begin + span, ng);
    for(int g = begin; g < end; ++g)
    {
        int64_t wgroup = weight_group_index<CODEC>(n, ng, g);
        float sw       = n < N ? group_scale<CODEC>(GS, WS, wgroup) : 1.f;
        float inv      = 1.f / sw;
        float wg       = (CODEC == 0 && n < N) ? WG[per_row ? n : 0] : 0.f;
        LaneGroup<CODEC> words{};
        if constexpr(is_vector<CODEC>)
            if(n < N)
                words.load(W, n, K, start + g * 128 + (lane / 16) * 4);
        I4 acc[MT] = {};
#pragma unroll
        for(int sub = 0; sub < 8; ++sub)
        {
            int lk = g * 128 + sub * 16 + (lane / 16) * 4, k = start + lk;
            uint32_t b = 0;
            if constexpr(is_vector<CODEC>)
                b = words.get(sub);
            else if(n < N && lk < L)
            {
                b = decode4<CODEC>(W, WS, GRID, n, K, k, wg, inv);
            }
#pragma unroll
   for(int t=0;t<MT;++t){
    int m=m0+t*16+lane%16;
    int a=(m<M&&lk<L)?*reinterpret_cast<const int*>(X+(int64_t)m*L+lk):0;
    acc[t]=__builtin_amdgcn_mfma_i32_16x16x16i8(a,(int)b,acc[t],0,0,0);
   }
  }
  #pragma unroll
  for(int t=0;t<MT;++t){
   #pragma unroll
   for(int j=0;j<4;++j){
    int m=m0+t*16+(lane/16)*4+j;
    float sx=m<M?XS[m*ng+g]:0.f;
    float dot = acc[t][j];
    if constexpr(is_w4<CODEC>)
        if(m < M && n < N)
            dot = affine_dot<CODEC>(acc[t][j], sums, (int64_t)m * ng + g, GRID, wgroup);
    accum[t][j] += (dot * sx) * sw;
   }
  }
 }
 #pragma unroll
 for(int t=0;t<MT;++t){
  #pragma unroll
  for(int j=0;j<4;++j){
   int m=m0+t*16+(lane/16)*4+j;
   if(m<M&&n<N)Y[(int64_t)blockIdx.z*M*N+(int64_t)m*N+n]=accum[t][j];
  }
 }
}
template <int CODEC>
void launch_codec(const int8_t* x,
                  const float* xs,
                  const uint8_t* w,
                  const uint8_t* ws,
                  const float* wg,
                  const float* gs,
                  const uint8_t* codebook,
                  float* y,
                  int m,
                  int n,
                  int k,
                  int start,
                  int length,
                  int per_row,
                  int split,
                  int mt,
                  hipStream_t stream,
                  const int32_t* sums = nullptr)
{
    dim3 grid((m + 16 * mt - 1) / (16 * mt), (n + 127) / 128, split);
    if(mt == 1)
        hipLaunchKernelGGL((packed_all<1, CODEC>),
                           grid,
                           dim3(512),
                           0,
                           stream,
                           x,
                           xs,
                           w,
                           ws,
                           wg,
                           gs,
                           codebook,
                           y,
                           m,
                           n,
                           k,
                           start,
                           length,
                           per_row,
                           split,
                           sums);
    else
        hipLaunchKernelGGL((packed_all<2, CODEC>),
                           grid,
                           dim3(512),
                           0,
                           stream,
                           x,
                           xs,
                           w,
                           ws,
                           wg,
                           gs,
                           codebook,
                           y,
                           m,
                           n,
                           k,
                           start,
                           length,
                           per_row,
                           split,
                           sums);
}
extern "C" int launch_packed(const int8_t* x,const float* xs,const uint8_t* w,
 const uint8_t* ws,const float* wg,const float* gs,const uint8_t* codebook,float* y,
 int m,int n,int k,int start,int length,int per_row,int split,int mt,int codec,hipStream_t stream){
 if(codec==0)launch_codec<0>(x,xs,w,ws,wg,gs,codebook,y,m,n,k,start,length,per_row,split,mt,stream);
 else if(codec==1)launch_codec<1>(x,xs,w,ws,wg,gs,codebook,y,m,n,k,start,length,per_row,split,mt,stream);
 else launch_codec<2>(x,xs,w,ws,wg,gs,codebook,y,m,n,k,start,length,per_row,split,mt,stream);
 return (int)hipGetLastError();
}

template <int MT, int CODEC, int NW, bool DOWN, bool ALIGNED, bool CACHED = false>
__global__ void packed_routed(const int8_t* X,
                              const float* XS,
                              const uint8_t* W,
                              const uint8_t* WS,
                              const float* WG,
                              const float* GS,
                              const uint8_t* GRID,
                              float* Y,
                              int M,
                              int N,
                              int K,
                              int start,
                              int L,
                              int per_row,
                              int split,
                              const int* SORT,
                              const int* EXP,
                              const int* COUNT,
                              int topk,
                              int down,
                              int64_t stride,
                              const int8_t* CACHE,
                              const int* MAP,
                              const int32_t* sums = nullptr)
{
    if(blockIdx.x * (16 * MT) >= *COUNT)
        return;
    int expert = EXP[blockIdx.x];
    int slot   = -1;
    if constexpr(CACHED)
        slot = MAP[expert];
    W += expert * stride;
    int64_t group_offset = (int64_t)expert * N * ((L + 127) / 128);
    if constexpr(is_w4<CODEC> || is_w8<CODEC>)
    {
        WS += group_offset * sizeof(__half);
        if constexpr(is_w4<CODEC>)
            GRID += group_offset * sizeof(__half);
    }
    else
        GS += group_offset;
    int lane = threadIdx.x % 64, wave = threadIdx.x / 64;
    int n  = blockIdx.y * (NW * 16) + wave * 16 + lane % 16;
    int m0 = blockIdx.x * (16 * MT);
    int input_row[MT], result_route[MT][4], result_row[MT][4];
    constexpr bool affine = is_w4<CODEC> || is_w8<CODEC>;
    if constexpr(affine)
    {
#pragma unroll
        for(int t = 0; t < MT; ++t)
        {
            int route    = SORT[m0 + t * 16 + lane % 16];
            int row      = DOWN ? route : (topk == 16 ? route >> 4 : route / topk);
            input_row[t] = route < M ? row : -1;
#pragma unroll
            for(int j = 0; j < 4; ++j)
            {
                int src            = (lane / 16) * 4 + j;
                result_route[t][j] = __shfl(route, src, 64);
                result_row[t][j]   = __shfl(row, src, 64);
            }
        }
    }
    float accum[MT][4] = {};
    int ng = (L + 127) / 128, span = (ng + split - 1) / split;
    int begin = blockIdx.z * span, end = min(begin + span, ng);
    for(int g = begin; g < end; ++g)
    {
        int64_t wgroup = weight_group_index<CODEC>(n, ng, g);
        float sw       = n < N ? group_scale<CODEC>(GS, WS, wgroup) : 1.f;
        float inv      = 1.f / sw;
        float wg       = (CODEC == 0 && n < N) ? WG[per_row ? n : 0] : 0.f;
        float d        = 0.f;
        if(CODEC == IQ2_XXS && DOWN && ALIGNED && n < N && slot < 0)
        {
            const uint8_t* base = W + (int64_t)n * (K / 256) * 66 + ((start + g * 128) / 256) * 66;
            d                   = __half2float(*reinterpret_cast<const __half*>(base));
        }
        LaneGroup<CODEC> words{};
        if constexpr(is_vector<CODEC>)
            if(n < N)
                words.load(W, n, K, start + g * 128 + (lane / 16) * 4);
        I4 acc[MT] = {};
#pragma unroll
        for(int sub = 0; sub < 8; ++sub)
        {
            int lk = g * 128 + sub * 16 + (lane / 16) * 4;
            int k  = ALIGNED ? (start / 128 + g) * 128 + sub * 16 + (lane / 16) * 4 : start + lk;
            uint32_t b = 0;
            if constexpr(is_vector<CODEC>)
                b = words.get(sub);
            else if(n < N && (ALIGNED || lk < L))
            {
                if constexpr(CACHED)
                {
                    if(slot >= 0)
                        b = *reinterpret_cast<const uint32_t*>(CACHE + ((int64_t)slot * N + n) * L +
                                                               lk);
                    else
                        b = decode4<CODEC, (DOWN && ALIGNED)>(W, WS, GRID, n, K, k, wg, inv, d);
                }
                else
                {
                    b = decode4<CODEC, (DOWN && ALIGNED)>(W, WS, GRID, n, K, k, wg, inv, d);
                }
            }
#pragma unroll
   for(int t=0;t<MT;++t){
       int xm;
       if constexpr(affine)
           xm = input_row[t];
       else
       {
           int route = SORT[m0 + t * 16 + lane % 16];
           xm        = route < M ? (DOWN ? route : route / topk) : -1;
       }
       int a  = (xm >= 0 && (ALIGNED || lk < L))
                    ? *reinterpret_cast<const int*>(X + (int64_t)xm * L + lk)
                    : 0;
       acc[t] = __builtin_amdgcn_mfma_i32_16x16x16i8(a, (int)b, acc[t], 0, 0, 0);
   }
  }
  #pragma unroll
  for(int t=0;t<MT;++t){
   #pragma unroll
   for(int j=0;j<4;++j){
       int route = affine ? result_route[t][j] : SORT[m0 + t * 16 + (lane / 16) * 4 + j];
       int xm    = affine ? result_row[t][j] : (DOWN ? route : route / topk);
       float sx  = route < M ? XS[xm * ng + g] : 0.f;
       float dot = acc[t][j];
       if constexpr(is_w4<CODEC>)
           if(route < M && n < N)
               dot = affine_dot<CODEC>(acc[t][j], sums, (int64_t)xm * ng + g, GRID, wgroup);
       accum[t][j] += (dot * sx) * sw;
   }
  }
 }
 #pragma unroll
 for(int t=0;t<MT;++t){
  #pragma unroll
  for(int j=0;j<4;++j){
      int route = affine ? result_route[t][j] : SORT[m0 + t * 16 + (lane / 16) * 4 + j];
      if(route < M && n < N)
          Y[(int64_t)blockIdx.z * M * N + (int64_t)route * N + n] = accum[t][j];
  }
 }
}

template<int NW,bool DOWN,bool ALIGNED,bool CACHED=false>
void launch_route_impl(const int8_t* x,const float* xs,const uint8_t* w,
 const float* gs,const uint8_t* codebook,float* y,const int* sorted,const int* experts,const int* count,
 int routes,int n,int k,int start,int length,int tiles,int topk,int64_t stride,hipStream_t stream,
 const int8_t* cache=nullptr,const int* map=nullptr){
 dim3 grid(tiles,(n+NW*16-1)/(NW*16),1);
 hipLaunchKernelGGL((packed_routed<1,2,NW,DOWN,ALIGNED,CACHED>),grid,dim3(NW*64),0,stream,x,xs,w,w,xs,gs,codebook,y,routes,n,k,start,length,0,1,sorted,experts,count,topk,DOWN,stride,cache,map);
}
extern "C" int launch_routed(const int8_t* x,const float* xs,const uint8_t* w,
 const float* gs,const uint8_t* codebook,float* y,const int* sorted,const int* experts,const int* count,
 int routes,int n,int k,int start,int length,int tiles,int topk,int down,int waves,int64_t stride,hipStream_t stream){
 #define ROUTE_IMPL(NW,D,A) launch_route_impl<NW,D,A>(x,xs,w,gs,codebook,y,sorted,experts,count,routes,n,k,start,length,tiles,topk,stride,stream)
 #define ROUTE(NW,D) do { if(start%128==0 && length%128==0){ROUTE_IMPL(NW,D,true);}else{ROUTE_IMPL(NW,D,false);} } while(0)
 if(waves==2){if(down){ROUTE(2,true);}else{ROUTE(2,false);}}
 else if(waves==4){if(down){ROUTE(4,true);}else{ROUTE(4,false);}}
 else {if(down){ROUTE(8,true);}else{ROUTE(8,false);}}
 #undef ROUTE
 #undef ROUTE_IMPL
 return (int)hipGetLastError();
}

extern "C" int launch_routed_cached(const int8_t* x,const float* xs,const uint8_t* w,
 const float* gs,const uint8_t* codebook,float* y,const int* sorted,const int* experts,const int* count,
 int routes,int n,int k,int start,int length,int tiles,int topk,int down,int waves,int64_t stride,
 const int8_t* cache,const int* map,hipStream_t stream){
 #define ROUTE_IMPL(NW,D,A) launch_route_impl<NW,D,A,true>(x,xs,w,gs,codebook,y,sorted,experts,count,routes,n,k,start,length,tiles,topk,stride,stream,cache,map)
 #define ROUTE(NW,D) do { if(start%128==0 && length%128==0){ROUTE_IMPL(NW,D,true);}else{ROUTE_IMPL(NW,D,false);} } while(0)
 if(waves==2){if(down){ROUTE(2,true);}else{ROUTE(2,false);}}
 else if(waves==4){if(down){ROUTE(4,true);}else{ROUTE(4,false);}}
 else {if(down){ROUTE(8,true);}else{ROUTE(8,false);}}
 #undef ROUTE
 #undef ROUTE_IMPL
 return (int)hipGetLastError();
}

// W4A8 uses two unsigned 4-bit activation planes. A third packed dot
// subtracts the +128 offset used to represent the signed activation as u8.
__device__ __forceinline__ unsigned nibble_plane(unsigned four_bytes)
{
    unsigned low   = four_bytes & 0x0f0f0f0f;
    unsigned pairs = (low | (low >> 4)) & 0x00ff00ff;
    return (pairs | (pairs >> 8)) & 0xffff;
}

template <int Codec>
__global__ void affine_dot8_gemv(const int8_t* x,
                                 const float* xs,
                                 const int32_t* sums,
                                 const uint8_t* w,
                                 const __half* scales,
                                 const __half* zeros,
                                 float* out,
                                 int m,
                                 int n,
                                 int k)
{
    int row = blockIdx.x * 16 + threadIdx.x / 16;
    if(row >= n)
        return;
    int lane = threadIdx.x % 16, token = blockIdx.y, groups = k / 128;
    float result = 0.f;
    for(int g = 0; g < groups; ++g)
    {
        int col = g * 128 + lane * 8;
        unsigned a[2], b;
        __builtin_memcpy(a, x + (int64_t)token * k + col, 8);
        if constexpr(Codec == OVERLAP_W4)
        {
            // Four bytes contain k0..3 in the low nibbles and k4..7 in the high.
            int64_t packet =
                (((int64_t)(row / 64) * groups + g) * 8 + (col % 128) / 16) * 64 + row % 64;
            unsigned pair;
            __builtin_memcpy(&pair, w + packet * 8 + (col % 16) / 8 * 4, 4);
            pair ^= 0x88888888u;
            b = nibble_plane(pair) | (nibble_plane(pair >> 4) << 16);
        }
        else if constexpr(is_wave<Codec>)
        {
            const auto* words = reinterpret_cast<const uint16_t*>(w);
            b                 = words[weight_word_index<Codec>(row, k, col)] |
                (uint32_t(words[weight_word_index<Codec>(row, k, col + 4)]) << 16);
        }
        else
            __builtin_memcpy(&b, w + ((int64_t)row * k + col) / 2, 4);
        a[0] ^= 0x80808080u;
        a[1] ^= 0x80808080u;
        unsigned lo = nibble_plane(a[0]) | (nibble_plane(a[1]) << 16);
        unsigned hi = nibble_plane(a[0] >> 4) | (nibble_plane(a[1] >> 4) << 16);
        int part    = (int)__builtin_amdgcn_udot8(b, lo, 0, false) +
                   16 * (int)__builtin_amdgcn_udot8(b, hi, 0, false) -
                   128 * (int)__builtin_amdgcn_udot8(b, 0x11111111u, 0, false);
#pragma unroll
        for(int shift = 8; shift > 0; shift >>= 1)
            part += __shfl_down(part, shift, 16);
        if(lane == 0)
        {
            int64_t wg = Codec == OVERLAP_W4 ? (int64_t)g * n + row
                                             : weight_group_index<Codec>(row, groups, g);
            int64_t ag = (int64_t)token * groups + g;
            result += ((float(part) - __half2float(zeros[wg]) * sums[ag]) * xs[ag]) *
                      __half2float(scales[wg]);
        }
    }
    if(lane == 0)
        out[(int64_t)token * n + row] = result;
}

template <int Codec, bool Down, int Waves>
void launch_affine_mfma(const int8_t* x,
                        const float* xs,
                        const int32_t* sums,
                        const uint8_t* w,
                        const uint8_t* ws,
                        const uint8_t* wz,
                        float* out,
                        const int* sorted,
                        const int* experts,
                        const int* count,
                        int m,
                        int n,
                        int k,
                        int tiles,
                        int topk,
                        int split,
                        int tile_m,
                        hipStream_t stream)
{
    if(!sorted)
    {
        launch_codec<Codec>(x,
                            xs,
                            w,
                            ws,
                            nullptr,
                            nullptr,
                            wz,
                            out,
                            m,
                            n,
                            k,
                            0,
                            k,
                            0,
                            split,
                            m <= 16 ? 1 : 2,
                            stream,
                            sums);
        return;
    }
    constexpr int waves = Waves;
    int64_t stride      = (int64_t)n * k / (is_w4<Codec> ? 2 : 1);
    dim3 grid(tiles, (n + waves * 16 - 1) / (waves * 16), 1);
#define HY4_MT(T)                                                    \
    hipLaunchKernelGGL((packed_routed<T, Codec, waves, Down, true>), \
                       grid,                                         \
                       dim3(waves * 64),                             \
                       0,                                            \
                       stream,                                       \
                       x,                                            \
                       xs,                                           \
                       w,                                            \
                       ws,                                           \
                       nullptr,                                      \
                       nullptr,                                      \
                       wz,                                           \
                       out,                                          \
                       m,                                            \
                       n,                                            \
                       k,                                            \
                       0,                                            \
                       k,                                            \
                       0,                                            \
                       1,                                            \
                       sorted,                                       \
                       experts,                                      \
                       count,                                        \
                       topk,                                         \
                       Down,                                         \
                       stride,                                       \
                       nullptr,                                      \
                       nullptr,                                      \
                       sums)
    if constexpr(is_vector<Codec>)
    {
        if(tile_m == 64)
        {
            HY4_MT(4);
            return;
        }
        if(tile_m == 32)
        {
            HY4_MT(2);
            return;
        }
    }
    HY4_MT(1);
#undef HY4_MT
}

template <int Codec>
void dispatch_affine(const int8_t* x,
                     const float* xs,
                     const int32_t* sums,
                     const uint8_t* w,
                     const uint8_t* ws,
                     const uint8_t* wz,
                     float* out,
                     const int* sorted,
                     const int* experts,
                     const int* count,
                     int m,
                     int n,
                     int k,
                     int tiles,
                     int topk,
                     int down,
                     int split,
                     int dot8,
                     int waves,
                     int tile_m,
                     hipStream_t stream)
{
    if constexpr(is_w4<Codec>)
    {
        if(dot8)
        {
            hipLaunchKernelGGL((affine_dot8_gemv<Codec>),
                               dim3((n + 15) / 16, m),
                               dim3(256),
                               0,
                               stream,
                               x,
                               xs,
                               sums,
                               w,
                               reinterpret_cast<const __half*>(ws),
                               reinterpret_cast<const __half*>(wz),
                               out,
                               m,
                               n,
                               k);
            return;
        }
    }
#define HY4_WAVES(D, W)                      \
    launch_affine_mfma<Codec, D, W>(x,       \
                                    xs,      \
                                    sums,    \
                                    w,       \
                                    ws,      \
                                    wz,      \
                                    out,     \
                                    sorted,  \
                                    experts, \
                                    count,   \
                                    m,       \
                                    n,       \
                                    k,       \
                                    tiles,   \
                                    topk,    \
                                    split,   \
                                    tile_m,  \
                                    stream)
    if(!sorted)
    {
        HY4_WAVES(false, 2);
    }
    else if(waves == 8)
    {
        if(down)
        {
            HY4_WAVES(true, 8);
        }
        else
        {
            HY4_WAVES(false, 8);
        }
    }
    else if(waves == 4)
    {
        if(down)
        {
            HY4_WAVES(true, 4);
        }
        else
        {
            HY4_WAVES(false, 4);
        }
    }
    else
    {
        if(down)
        {
            HY4_WAVES(true, 2);
        }
        else
        {
            HY4_WAVES(false, 2);
        }
    }
#undef HY4_WAVES
}

extern "C" int launch_affine(const int8_t* x,
                             const float* xs,
                             const int32_t* sums,
                             const uint8_t* w,
                             const uint8_t* ws,
                             const uint8_t* wz,
                             float* out,
                             const int* sorted,
                             const int* experts,
                             const int* count,
                             int m,
                             int n,
                             int k,
                             int tiles,
                             int topk,
                             int down,
                             int split,
                             int dot8,
                             int wave_format,
                             int waves,
                             int tile_m,
                             hipStream_t stream)
{
    if(wave_format == 3)
    {
        if(!dot8 || !wz || sorted || split != 1)
            return hipErrorInvalidValue;
        hipLaunchKernelGGL((affine_dot8_gemv<OVERLAP_W4>),
                           dim3((n + 15) / 16, m),
                           dim3(256),
                           0,
                           stream,
                           x,
                           xs,
                           sums,
                           w,
                           reinterpret_cast<const __half*>(ws),
                           reinterpret_cast<const __half*>(wz),
                           out,
                           m,
                           n,
                           k);
        return hipGetLastError();
    }
#define HY4_CODEC(C)            \
    dispatch_affine<C>(x,       \
                       xs,      \
                       sums,    \
                       w,       \
                       ws,      \
                       wz,      \
                       out,     \
                       sorted,  \
                       experts, \
                       count,   \
                       m,       \
                       n,       \
                       k,       \
                       tiles,   \
                       topk,    \
                       down,    \
                       split,   \
                       dot8,    \
                       waves,   \
                       tile_m,  \
                       stream)
    if(wz)
    {
        if(wave_format == 2)
        {
            HY4_CODEC(VECTOR_W4);
        }
        else if(wave_format)
        {
            HY4_CODEC(WAVE_W4);
        }
        else
        {
            HY4_CODEC(AFFINE_W4);
        }
    }
    else
    {
        if(wave_format == 2)
        {
            HY4_CODEC(VECTOR_W8);
        }
        else if(wave_format)
        {
            HY4_CODEC(WAVE_W8);
        }
        else
        {
            HY4_CODEC(SIGNED_W8);
        }
    }
#undef HY4_CODEC
    return (int)hipGetLastError();
}
