#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#endif

void unload_matmul_fp16_36f7d82d_0d1d2d34567c89c10d11c(void);
void load_matmul_fp16_36f7d82d_0d1d2d34567c89c10d11c(void);
// tt-linker: matmul_fp16_36f7d82d_0d1d2d34567c89c10d11c:CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn, int32_t stride_am, int32_t stride_ak, int32_t stride_bk, int32_t stride_bn:16x16x16_warps1xstages3
CUresult matmul_fp16_36f7d82d_0d1d2d34567c89c10d11c(CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M, int32_t N, int32_t K, int32_t stride_cm, int32_t stride_am, int32_t stride_bk);
