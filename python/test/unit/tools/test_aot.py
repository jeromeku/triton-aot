import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import triton
from triton.common import cuda_include_dir, libcuda_dirs

add_kernel_src = """
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    N,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)
"""

kernel_utils_src = """
import triton

@triton.jit
def mul(x, y):
    return x * y
"""

kernel_src = """
import triton
import triton.language as tl
import kernel_utils

@triton.jit
def kernel(C, A, B, M, N, K,
          stride_cm, stride_cn,
          stride_am, stride_ak,
          stride_bk, stride_bn,
          BLOCK_M: tl.constexpr,
          BLOCK_N: tl.constexpr,
          BLOCK_K: tl.constexpr):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
  offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
  offs_k = tl.arange(0, BLOCK_K)
  a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

  accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  for k in range(0, tl.cdiv(K, BLOCK_K)):
      # Load the next block of A and B, generate a mask by checking the K dimension.
      # If it is out of bounds, set it to 0.
      a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
      b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
      # We accumulate along the K dimension.
      accumulator += tl.dot(a, b)
      # Advance the ptrs to the next K block.
      a_ptrs += BLOCK_K * stride_ak
      b_ptrs += BLOCK_K * stride_bk

  c = kernel_utils.mul(accumulator, accumulator)
  # Write back the block of the output matrix C with masks.
  offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  tl.store(c_ptrs, c)
"""

test_utils_src = """
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "kernel.h"

static void write_buffer_to_csv(char *filename, int32_t *buffer, int size) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    for (int i = 0; i < size; i++) {
        fprintf(file, "%d", buffer[i]);
        if (i < size - 1) {
            fprintf(file, ",");
        }
    }
    fclose(file);
}

static void read_csv_to_buffer(char *filename, int16_t *buffer, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    int index = 0;
    while (fscanf(file, "%hd,", &buffer[index]) != EOF && index < size) {
        index++;
    }
    fclose(file);
}"""


def gen_kernel_library(dir, libname):
    c_files = glob.glob(os.path.join(dir, "*.c"))
    subprocess.run(
        ["gcc"] + c_files + ["-I", cuda_include_dir(), "-c", "-fPIC"],
        check=True,
        cwd=dir,
    )
    o_files = glob.glob(os.path.join(dir, "*.o"))
    subprocess.run(
        ["gcc"] + o_files + ["-shared", "-o", libname, "-L", libcuda_dirs()[0]],
        check=True,
        cwd=dir,
    )


def gen_test_bin(dir, M, N, K, exe="test", algo_id=0):
    test_src = f"""
int main(int argc, char **argv) {{
  int M = {M}, N = {N}, K = {K};

  // initialize CUDA handles
  CUdevice dev;
  CUcontext ctx;
  CUstream stream;
  CUdeviceptr A, B, C;
  CUresult err = 0;
  cuInit(0);
  cuDeviceGet(&dev, 0);
  cuCtxCreate(&ctx, 0, dev);
  cuMemAlloc(&A, M * K * 2);
  cuMemAlloc(&B, K * N * 2);
  cuMemAlloc(&C, M * N * 4);
  cuStreamCreate(&stream, 0);
  load_matmul_fp16();

  // initialize input data
  int16_t hA[M*K];
  int16_t hB[K*N];
  memset(hA, 0, M*K*2);
  memset(hB, 0, K*N*2);
  read_csv_to_buffer(argv[1], hA, M*K);
  read_csv_to_buffer(argv[2], hB, K*N);
  cuMemcpyHtoD(A, hA, M*K*2);
  cuMemcpyHtoD(B, hB, K*N*2);

  // launch kernel
  cuStreamSynchronize(stream);
  CUresult ret;
  int algo_id = {algo_id};
  if (algo_id == 0) {{
    ret = matmul_fp16_default(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1);
  }} else {{
    ret = matmul_fp16(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1, {algo_id});
  }}
  if (ret != 0) fprintf(stderr, "kernel launch failed\\n");
  assert(ret == 0);

  cuStreamSynchronize(stream);

  // read data
  int32_t hC[M*N];
  memset(hC, 0, M*N*4);
  cuMemcpyDtoH(hC, C, M*N*4);
  write_buffer_to_csv(argv[3], hC, M*N);

  // free cuda handles
  unload_matmul_fp16();
  cuMemFree(A);
  cuMemFree(B);
  cuMemFree(C);
  cuCtxDestroy(ctx);
}}
"""
    src = test_utils_src + test_src
    with open(os.path.join(dir, "test.c"), "w") as file:
        file.write(src)
    subprocess.run(
        ["gcc"]
        + [
            "test.c",
            "-I",
            cuda_include_dir(),
            "-L",
            libcuda_dirs()[0],
            "-l",
            "cuda",
            "-L",
            dir,
            "-l",
            "kernel",
            "-o",
            exe,
        ],
        check=True,
        cwd=dir,
    )


def gen_add_test_bin(
    dir,
    N,
    kernel_name,
    dtype_in,
    dtype_out,
    kernel_lib_name=None,
    exe="test",
    algo_id=0,
):
    if "16" in dtype_in:
        num_bytes_in = 2
        in_fmt_str = "%hd"
    elif "32" in dtype_in:
        num_bytes_in = 4
        in_fmt_str = "%d"

    if "16" in dtype_out:
        num_bytes_out = 2
        out_fmt_str = "%hd"

    elif "32" in dtype_out:
        num_bytes_out = 4
        out_fmt_str = "%d"

    headers = f"""
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "{kernel_name}.h"

"""

    test_utils_src = (
        headers
        + """
static void write_buffer_to_csv(char *filename, {dtype_out} *buffer, int size) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    printf("Writing to %s\\n", filename);
    for (int i = 0; i < size; i++) {
        fprintf(file, "{out_fmt_str}", buffer[i]);
        if (i < size - 1) {
            fprintf(file, ",");
        }
    }
    fclose(file);
}

static void read_csv_to_buffer(char *filename, {dtype_in} *buffer, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    int index = 0;
    printf("Reading from %s\\n", filename);
    while (fscanf(file, "{in_fmt_str},", &buffer[index]) != EOF && index < size) {
        index++;
    }
    fclose(file);
}""".format(
            in_fmt_str=in_fmt_str,
            out_fmt_str=out_fmt_str,
            dtype_in=dtype_in,
            dtype_out=dtype_out,
        )
    )

    test_src = f"""
int main(int argc, char **argv) {{
  int N = {N};

  // initialize CUDA handles
  CUdevice dev;
  CUcontext ctx;
  CUstream stream;
  CUdeviceptr x, y, out;
  CUresult err = 0;
  cuInit(0);
  cuDeviceGet(&dev, 0);
  cuCtxCreate(&ctx, 0, dev);
  cuMemAlloc(&x, N);
  cuMemAlloc(&y, N);
  cuMemAlloc(&out, N);
  cuStreamCreate(&stream, 0);
  load_add_kernel();

  // initialize input data
  int16_t hx[N];
  int16_t hy[N];
  memset(hx, 0, N * {num_bytes_in});
  memset(hy, 0, N * {num_bytes_in});
  read_csv_to_buffer(argv[1], hx, N);
  read_csv_to_buffer(argv[2], hy, N);
  cuMemcpyHtoD(x, hx, N * {num_bytes_in});
  cuMemcpyHtoD(y, hy, N * {num_bytes_in});

  // launch kernel
  cuStreamSynchronize(stream);
  CUresult ret;
  int algo_id = {algo_id};
  if (algo_id == 0) {{
    ret = add_kernel_default(stream, x, y, out, N);
  }} else {{
    ret = add_kernel(stream, x, y, out, N, {algo_id});
  }}
  if (ret != 0) fprintf(stderr, "kernel launch failed\\n");
  assert(ret == 0);

  cuStreamSynchronize(stream);

  // read data
  int16_t hout[N];
  memset(hout, 0, N);
  cuMemcpyDtoH(hout, out, N);
  write_buffer_to_csv(argv[3], hout, N);

  // free cuda handles
  unload_add_kernel();
  cuMemFree(x);
  cuMemFree(y);
  cuMemFree(out);
  cuCtxDestroy(ctx);
}}
"""
    kernel_lib_name = kernel_lib_name or kernel_name
    src = test_utils_src + test_src
    with open(os.path.join(dir, f"{exe}.c"), "w") as file:
        file.write(src)
    subprocess.run(
        ["gcc"]
        + [
            f"{exe}.c",
            "-I",
            cuda_include_dir(),
            "-L",
            libcuda_dirs()[0],
            "-l",
            "cuda",
            "-L",
            dir,
            "-l",
            kernel_lib_name,
            "-o",
            exe,
        ],
        check=True,
        cwd=dir,
    )


def write_triton_kernels(dir, src, util_src):
    kernel_path = os.path.join(dir, "kernel.py")
    with open(kernel_path, "w") as file:
        file.write(src)

    kernel_utils_path = os.path.join(dir, "kernel_utils.py")
    with open(kernel_utils_path, "w") as file:
        file.write(util_src)

    return kernel_path


def write_tt_kernel(dir, src, name):
    kernel_path = os.path.join(dir, name)
    with open(kernel_path, "w") as file:
        file.write(src)

    return Path(kernel_path).absolute()


def _find_kernel_name(kernel_path):
    import ast

    with open(kernel_path) as fp:
        tree = ast.parse(fp.read())
        fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        assert len(fns) == 1
    return fns[0].name


def _dtype_map(ty):
    return {np.int16: "i16", np.int32: "i32", np.float16: "fp16", np.float32: "fp32"}[
        ty
    ]


def compile_kernel_add(
    dir,
    kernel_path,
    N,
    dtype=np.float16,
    BLOCK_SIZE=1024,
    name=None,
    num_warps=1,
    specializations=None,
):
    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")
    # x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr

    sig = f"*{_dtype_map(dtype)}, *{_dtype_map(dtype)}, *{_dtype_map(dtype)}, i32, {BLOCK_SIZE}"
    name = name or _find_kernel_name(kernel_path)
    grid = f"N/{BLOCK_SIZE}, 1, 1"
    num_warps = str(num_warps)
    subprocess.run(
        [
            sys.executable,
            compiler_path,
            "-n",
            name,
            "--signature",
            sig,
            "--out-name",
            name,
            "-o",
            name,
            "-w",
            num_warps,
            "-g",
            grid,
            kernel_path,
        ],
        check=True,
        cwd=dir,
    )


def compile_aot_kernels(dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints):
    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

    # compile all desired configs
    for ha in ha_hb_hints:
        for hb in ha_hb_hints:
            sig = f"*fp32:16, *{dtype}:16, *{dtype}:16, i32, i32, i32, i32{ha}, i32:1, i32{hb}, i32:1, i32:16, i32:1, {BM}, {BN}, {BK}"
            name = f"matmul_{dtype}"
            grid = f"M/{BM}, N/{BN}, 1"
            subprocess.run(
                [
                    sys.executable,
                    compiler_path,
                    "-n",
                    "kernel",
                    "--signature",
                    sig,
                    "--out-name",
                    name,
                    "-o",
                    name,
                    "-w",
                    "1",
                    "-g",
                    grid,
                    kernel_path,
                ],
                check=True,
                cwd=dir,
            )


def link_aot_kernels(dir, out_name="kernel"):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # link all desired configs
    h_files = glob.glob(os.path.join(dir, "*.h"))
    subprocess.run(
        [sys.executable, linker_path] + h_files + ["-o", out_name], check=True, cwd=dir
    )


def _link_aot_kernels(dir, out_name):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # link all desired configs
    h_files = glob.glob(os.path.join(dir, "*.h"))
    subprocess.run(
        [sys.executable, linker_path] + h_files + ["-o", out_name], check=True, cwd=dir
    )


def generate_matmul_test_data(dir, M, N, K):
    a = np.random.randn(M * K).astype(np.float16).reshape((M, K))
    b = np.random.randn(M * K).astype(np.float16).reshape((K, N))
    a_path = os.path.join(dir, "a.csv")
    b_path = os.path.join(dir, "b.csv")
    c_path = os.path.join(dir, "c.csv")
    for x, path in [(a, a_path), (b, b_path)]:
        x.view(np.int16).ravel().tofile(path, sep=",")
    return a, b, a_path, b_path, c_path


def generate_test_data(dir, shape, file_name, dtype=np.float32, seed=0, ext="csv"):
    x = np.random.randn(np.prod(shape)).astype(dtype).reshape(shape)
    x_path = os.path.join(dir, f"{file_name}.{ext}")
    x.ravel().tofile(x_path, sep=",")
    return x, x_path


def generate_dummy_data(dir, shape, file_name, dtype=np.float16, seed=0, ext="csv"):
    x = np.ones(np.prod(shape)).astype(dtype).reshape(shape)
    x_path = os.path.join(dir, f"{file_name}.{ext}")
    x.view(np.int16).ravel().tofile(x_path, sep=",")
    return x, x_path


def check_dir(dir):
    if os.path.exists(dir):
        import shutil

        shutil.rmtree(dir)

    os.makedirs(dir)
    return dir


def test_compile_link_add():
    from pathlib import Path

    N = 1024
    BLOCK_SIZE = 1024
    NUM_WARPS = 4
    kernel_dir = Path("aot_kernels").absolute()
    check_dir(kernel_dir)
    dtype = np.int16
    kernel_path = write_tt_kernel(kernel_dir, add_kernel_src, "add_kernel.py")
    kernel_name = _find_kernel_name(kernel_path)
    compile_kernel_add(
        kernel_dir,
        kernel_path,
        N,
        name=kernel_name,
        dtype=dtype,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NUM_WARPS,
    )
    link_aot_kernels(kernel_dir, out_name=kernel_name)
    executable_name = "test"
    gen_kernel_library(kernel_dir, f"lib{kernel_name}.so")
    gen_add_test_bin(
        kernel_dir,
        N,
        kernel_name=kernel_name,
        exe=executable_name,
    )

    # Generate test data
    seed = 0
    data_dir = Path("test_data").absolute()
    check_dir(data_dir)

    data_generator = generate_dummy_data
    x, x_path = data_generator(data_dir, (N,), file_name="x", seed=seed, dtype=dtype)
    y, y_path = data_generator(data_dir, (N,), file_name="y", seed=seed, dtype=dtype)
    out_path = os.path.join(data_dir, "out.csv")
    expected = x + y
    # print(f"EXPECTED: {expected}")

    # run test case
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = kernel_dir

    subprocess.run(
        [f"./{executable_name}", x_path, y_path, out_path],
        env=env,
        check=True,
        cwd=kernel_dir,
    )

    # read data and compare against reference
    actual = np.genfromtxt(out_path, delimiter=",", dtype=dtype)
    EXPECTED_VAL = 2.0

    def compute_stats(x):
        actual_counts = np.isclose(x, EXPECTED_VAL).sum()
        return actual_counts

    print(f"ACTUAL counts: {compute_stats(actual)}")

    # c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
    # np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


def test_compile_link_matmul():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernels(
            tmp_dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints=["", ":16"]
        )
        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library(tmp_dir, "libkernel.so")
        gen_test_bin(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = tmp_dir
        subprocess.run(
            ["./test", a_path, b_path, c_path], env=env, check=True, cwd=tmp_dir
        )

        # read data and compare against reference
        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((M, N)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


def test_launcher_has_no_available_kernel():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernels(tmp_dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints=[":1"])
        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library(tmp_dir, "libkernel.so")
        gen_test_bin(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = tmp_dir
        result = subprocess.run(
            ["./test", a_path, b_path, c_path],
            env=env,
            cwd=tmp_dir,
            capture_output=True,
            text=True,
        )

        # It should fail since the launcher requires all the strides be 1 while they are not.
        assert result.returncode == -6
        assert "kernel launch failed" in result.stderr


def test_compile_link_autotune_matmul():
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)

        tile_sizes = [
            [16, 16, 16],
            [32, 32, 16],
            [32, 32, 32],
            [64, 64, 32],
        ]

        for ts in tile_sizes:
            BM, BN, BK = ts[0], ts[1], ts[2]
            compile_aot_kernels(
                tmp_dir, kernel_path, dtype, BM, BN, BK, ha_hb_hints=["", ":16"]
            )

        link_aot_kernels(tmp_dir)

        gen_kernel_library(tmp_dir, "libkernel.so")

        # compile test case
        M, N, K = 64, 64, 64
        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))

        for algo_id in range(len(tile_sizes)):
            # generate and run test case
            test_name = f"test_{algo_id}"
            gen_test_bin(tmp_dir, M, N, K, exe=test_name, algo_id=algo_id)

            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = tmp_dir
            subprocess.run(
                [f"./{test_name}", a_path, b_path, c_path],
                check=True,
                cwd=tmp_dir,
                env=env,
            )

            # read data and compare against reference
            c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
            c_tri = c.reshape((M, N)).view(np.float32)
            np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=1e-4)


def test_ttgir_to_ptx():
    src = """
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32, "triton_gpu.num-ctas" = 1 : i32} {
  tt.func public @sum_kernel_0d1d(%arg0: !tt.ptr<i32, 1>, %arg1: !tt.ptr<i32, 1>) {
    tt.return
  }
}
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = os.path.join(tmp_dir, "empty_kernel.ttgir")
        with open(kernel_path, "w") as fp:
            fp.write(src)
        k = triton.compile(kernel_path, cc=80)
        ptx = k.asm["ptx"]
        assert ".target sm_80" in ptx
        assert ".address_size 64" in ptx
        assert ".address_size 64" in ptx
