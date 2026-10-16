// SPDX-License-Identifier: Apache-2.0
// Register-local packed-weight to GS128 INT8 conversion and gfx908 MFMA.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <hip/hip_fp16.h>
using I4 = int __attribute__((ext_vector_type(4)));
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
 }else{
  const uint8_t* base=W+(int64_t)n*(K/256)*66+(k/256)*66;
  float d=HINT?d_hint:__half2float(*reinterpret_cast<const __half*>(base));
  int sub=(k%256)/32,lane=(k%32)/8,j0=k%8;
  int code=base[2+sub*8+lane];
  uint32_t bits;__builtin_memcpy(&bits,base+6+sub*8,4);
  int sign7=(bits>>(7*lane))&127,signs=sign7|((__popc(sign7)&1)<<7);
  float scale=(d*(0.5f+(float)(bits>>28)))*0.25f;
  uint32_t values;__builtin_memcpy(&values,GRID+code*8+j0,4);
  #pragma unroll
  for(int j=0;j<4;++j){
   float v=scale*(float)((values>>(j*8))&255);
   if((signs>>(j0+j))&1)v=-v;
   result|=(uint32_t)(uint8_t)(int8_t)__float2int_rn(v*inv)<<(j*8);
  }
 }
 return result;
}
template<int MT,int CODEC>
__global__ void packed_all(const int8_t* X,const float* XS,const uint8_t* W,
 const uint8_t* WS,const float* WG,const float* GS,const uint8_t* GRID,float* Y,
 int M,int N,int K,int start,int L,int per_row,int split) {
 int lane=threadIdx.x%64,wave=threadIdx.x/64;
 int n=blockIdx.y*128+wave*16+lane%16;
 int m0=blockIdx.x*(16*MT);
 float accum[MT][4]={};
 int ng=(L+127)/128,span=(ng+split-1)/split;
 int begin=blockIdx.z*span,end=min(begin+span,ng);
 for(int g=begin;g<end;++g){
  float sw=n<N?GS[n*ng+g]:1.f;
  float inv=1.f/sw;
  float wg=(CODEC==0&&n<N)?WG[per_row?n:0]:0.f;
  I4 acc[MT]={};
  #pragma unroll
  for(int sub=0;sub<8;++sub){
   int lk=g*128+sub*16+(lane/16)*4,k=start+lk;
   uint32_t b=0;
   if(n<N&&lk<L){
    b=decode4<CODEC>(W,WS,GRID,n,K,k,wg,inv);
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
    accum[t][j]+=((float)acc[t][j]*sx)*sw;
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
template<int CODEC>
void launch_codec(const int8_t* x,const float* xs,const uint8_t* w,
 const uint8_t* ws,const float* wg,const float* gs,const uint8_t* codebook,float* y,
 int m,int n,int k,int start,int length,int per_row,int split,int mt,hipStream_t stream){
 dim3 grid((m+16*mt-1)/(16*mt),(n+127)/128,split);
 if(mt==1)hipLaunchKernelGGL((packed_all<1,CODEC>),grid,dim3(512),0,stream,x,xs,w,ws,wg,gs,codebook,y,m,n,k,start,length,per_row,split);
 else hipLaunchKernelGGL((packed_all<2,CODEC>),grid,dim3(512),0,stream,x,xs,w,ws,wg,gs,codebook,y,m,n,k,start,length,per_row,split);
}
extern "C" int launch_packed(const int8_t* x,const float* xs,const uint8_t* w,
 const uint8_t* ws,const float* wg,const float* gs,const uint8_t* codebook,float* y,
 int m,int n,int k,int start,int length,int per_row,int split,int mt,int codec,hipStream_t stream){
 if(codec==0)launch_codec<0>(x,xs,w,ws,wg,gs,codebook,y,m,n,k,start,length,per_row,split,mt,stream);
 else if(codec==1)launch_codec<1>(x,xs,w,ws,wg,gs,codebook,y,m,n,k,start,length,per_row,split,mt,stream);
 else launch_codec<2>(x,xs,w,ws,wg,gs,codebook,y,m,n,k,start,length,per_row,split,mt,stream);
 return (int)hipGetLastError();
}

template<int MT,int CODEC,int NW,bool DOWN,bool ALIGNED,bool CACHED=false>
__global__ void packed_routed(const int8_t* X,const float* XS,const uint8_t* W,
 const uint8_t* WS,const float* WG,const float* GS,const uint8_t* GRID,float* Y,
 int M,int N,int K,int start,int L,int per_row,int split,
 const int* SORT,const int* EXP,const int* COUNT,int topk,int down,int64_t stride,
 const int8_t* CACHE,const int* MAP) {
 if(blockIdx.x*16>=*COUNT)return;
 int expert=EXP[blockIdx.x];
 int slot=-1;
 if constexpr(CACHED)slot=MAP[expert];
 W+=expert*stride;
 GS+=(int64_t)expert*N*((L+127)/128);
 int lane=threadIdx.x%64,wave=threadIdx.x/64;
 int n=blockIdx.y*(NW*16)+wave*16+lane%16;
 int m0=blockIdx.x*(16*MT);
 float accum[MT][4]={};
 int ng=(L+127)/128,span=(ng+split-1)/split;
 int begin=blockIdx.z*span,end=min(begin+span,ng);
 for(int g=begin;g<end;++g){
  float sw=n<N?GS[n*ng+g]:1.f;
  float inv=1.f/sw;
  float wg=(CODEC==0&&n<N)?WG[per_row?n:0]:0.f;
  float d=0.f;
  if(DOWN && ALIGNED && n<N && slot<0){
   const uint8_t* base=W+(int64_t)n*(K/256)*66+((start+g*128)/256)*66;
   d=__half2float(*reinterpret_cast<const __half*>(base));
  }
  I4 acc[MT]={};
  #pragma unroll
  for(int sub=0;sub<8;++sub){
   int lk=g*128+sub*16+(lane/16)*4;
   int k=ALIGNED?(start/128+g)*128+sub*16+(lane/16)*4:start+lk;
   uint32_t b=0;
   if(n<N && (ALIGNED || lk<L)){
    if constexpr(CACHED){
     if(slot>=0)b=*reinterpret_cast<const uint32_t*>(CACHE+((int64_t)slot*N+n)*L+lk);
     else b=decode4<CODEC,(DOWN && ALIGNED)>(W,WS,GRID,n,K,k,wg,inv,d);
    }else{
     b=decode4<CODEC,(DOWN && ALIGNED)>(W,WS,GRID,n,K,k,wg,inv,d);
    }
   }
   #pragma unroll
   for(int t=0;t<MT;++t){
    int m=m0+t*16+lane%16;
    int route=SORT[m];
    int xm=DOWN?route:route/topk;
    int a=(route<M && (ALIGNED || lk<L))?*reinterpret_cast<const int*>(X+(int64_t)xm*L+lk):0;
    acc[t]=__builtin_amdgcn_mfma_i32_16x16x16i8(a,(int)b,acc[t],0,0,0);
   }
  }
  #pragma unroll
  for(int t=0;t<MT;++t){
   #pragma unroll
   for(int j=0;j<4;++j){
    int m=m0+t*16+(lane/16)*4+j;
    int route=SORT[m];
    int xm=DOWN?route:route/topk;
    float sx=route<M?XS[xm*ng+g]:0.f;
    accum[t][j]+=((float)acc[t][j]*sx)*sw;
   }
  }
 }
 #pragma unroll
 for(int t=0;t<MT;++t){
  #pragma unroll
  for(int j=0;j<4;++j){
   int m=m0+t*16+(lane/16)*4+j;
   int route=SORT[m];
   if(route<M&&n<N)Y[(int64_t)blockIdx.z*M*N+(int64_t)route*N+n]=accum[t][j];
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
