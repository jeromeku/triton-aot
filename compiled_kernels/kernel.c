#include <assert.h>
#include <cuda.h>
#include <stdint.h>

// launcher for: matmul_fp16_16x16x16_warps1xstages3
CUresult matmul_fp16_eda2ecb4_0d1d2d3456d7c8d9c10d11c(
    CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M,
    int32_t N, int32_t K, int32_t stride_cm, int32_t stride_am,
    int32_t stride_bk);
CUresult matmul_fp16_664515af_0d1d2d3456d7c89c10d11c(
    CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M,
    int32_t N, int32_t K, int32_t stride_cm, int32_t stride_am,
    int32_t stride_bk);
CUresult matmul_fp16_75e64dbd_0d1d2d34567c8d9c10d11c(
    CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M,
    int32_t N, int32_t K, int32_t stride_cm, int32_t stride_am,
    int32_t stride_bk);
CUresult matmul_fp16_36f7d82d_0d1d2d34567c89c10d11c(
    CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M,
    int32_t N, int32_t K, int32_t stride_cm, int32_t stride_am,
    int32_t stride_bk);

CUresult matmul_fp16_16x16x16_warps1xstages3(
    CUstream stream, CUdeviceptr C, CUdeviceptr A, CUdeviceptr B, int32_t M,
    int32_t N, int32_t K, int32_t stride_cm, int32_t stride_cn,
    int32_t stride_am, int32_t stride_ak, int32_t stride_bk,
    int32_t stride_bn) {
  if ((C % 16 == 0) && (A % 16 == 0) && (B % 16 == 0) &&
      (stride_cm % 16 == 0) && (stride_cn == 1) && (stride_am % 16 == 0) &&
      (stride_ak == 1) && (stride_bk % 16 == 0) && (stride_bn == 1))
    return matmul_fp16_eda2ecb4_0d1d2d3456d7c8d9c10d11c(
        stream, C, A, B, M, N, K, stride_cm, stride_am, stride_bk);
  if ((C % 16 == 0) && (A % 16 == 0) && (B % 16 == 0) &&
      (stride_cm % 16 == 0) && (stride_cn == 1) && (stride_ak == 1) &&
      (stride_bk % 16 == 0) && (stride_bn == 1))
    return matmul_fp16_664515af_0d1d2d3456d7c89c10d11c(
        stream, C, A, B, M, N, K, stride_cm, stride_am, stride_bk);
  if ((C % 16 == 0) && (A % 16 == 0) && (B % 16 == 0) && (stride_cn == 1) &&
      (stride_am % 16 == 0) && (stride_ak == 1) && (stride_bk % 16 == 0) &&
      (stride_bn == 1))
    return matmul_fp16_75e64dbd_0d1d2d34567c8d9c10d11c(
        stream, C, A, B, M, N, K, stride_cm, stride_am, stride_bk);
  if ((C % 16 == 0) && (A % 16 == 0) && (B % 16 == 0) && (stride_cn == 1) &&
      (stride_ak == 1) && (stride_bk % 16 == 0) && (stride_bn == 1))
    return matmul_fp16_36f7d82d_0d1d2d34567c89c10d11c(
        stream, C, A, B, M, N, K, stride_cm, stride_am, stride_bk);

  return CUDA_ERROR_INVALID_VALUE;
}

// load for: matmul_fp16_16x16x16_warps1xstages3
void load_matmul_fp16_eda2ecb4_0d1d2d3456d7c8d9c10d11c();
void load_matmul_fp16_664515af_0d1d2d3456d7c89c10d11c();
void load_matmul_fp16_75e64dbd_0d1d2d34567c8d9c10d11c();
void load_matmul_fp16_36f7d82d_0d1d2d34567c89c10d11c();
void load_matmul_fp16_16x16x16_warps1xstages3() {
  load_matmul_fp16_eda2ecb4_0d1d2d3456d7c8d9c10d11c();
  load_matmul_fp16_664515af_0d1d2d3456d7c89c10d11c();
  load_matmul_fp16_75e64dbd_0d1d2d34567c8d9c10d11c();
  load_matmul_fp16_36f7d82d_0d1d2d34567c89c10d11c();
}

// unload for: matmul_fp16_16x16x16_warps1xstages3
void unload_matmul_fp16_eda2ecb4_0d1d2d3456d7c8d9c10d11c();
void unload_matmul_fp16_664515af_0d1d2d3456d7c89c10d11c();
void unload_matmul_fp16_75e64dbd_0d1d2d34567c8d9c10d11c();
void unload_matmul_fp16_36f7d82d_0d1d2d34567c89c10d11c();
void unload_matmul_fp16_16x16x16_warps1xstages3() {
  unload_matmul_fp16_eda2ecb4_0d1d2d3456d7c8d9c10d11c();
  unload_matmul_fp16_664515af_0d1d2d3456d7c89c10d11c();
  unload_matmul_fp16_75e64dbd_0d1d2d34567c8d9c10d11c();
  unload_matmul_fp16_36f7d82d_0d1d2d34567c89c10d11c();
}

typedef CUresult (*kernel_func_t)(CUstream stream, CUdeviceptr C, CUdeviceptr A,
                                  CUdeviceptr B, int32_t M, int32_t N,
                                  int32_t K, int32_t stride_cm,
                                  int32_t stride_cn, int32_t stride_am,
                                  int32_t stride_ak, int32_t stride_bk,
                                  int32_t stride_bn);
kernel_func_t matmul_fp16_kernels[] = {
    matmul_fp16_16x16x16_warps1xstages3,
};

int matmul_fp16_get_num_algos(void) { return (int)sizeof(matmul_fp16_kernels); }

CUresult matmul_fp16(CUstream stream, CUdeviceptr C, CUdeviceptr A,
                     CUdeviceptr B, int32_t M, int32_t N, int32_t K,
                     int32_t stride_cm, int32_t stride_cn, int32_t stride_am,
                     int32_t stride_ak, int32_t stride_bk, int32_t stride_bn,
                     int algo_id) {
  assert(algo_id < (int)sizeof(matmul_fp16_kernels));
  return matmul_fp16_kernels[algo_id](stream, C, A, B, M, N, K, stride_cm,
                                      stride_cn, stride_am, stride_ak,
                                      stride_bk, stride_bn);
}

void load_matmul_fp16(void) { load_matmul_fp16_16x16x16_warps1xstages3(); }

void unload_matmul_fp16(void) { unload_matmul_fp16_16x16x16_warps1xstages3(); }

CUresult matmul_fp16_default(CUstream stream, CUdeviceptr C, CUdeviceptr A,
                             CUdeviceptr B, int32_t M, int32_t N, int32_t K,
                             int32_t stride_cm, int32_t stride_cn,
                             int32_t stride_am, int32_t stride_ak,
                             int32_t stride_bk, int32_t stride_bn) {
  return matmul_fp16(stream, C, A, B, M, N, K, stride_cm, stride_cn, stride_am,
                     stride_ak, stride_bk, stride_bn, 0);
}
