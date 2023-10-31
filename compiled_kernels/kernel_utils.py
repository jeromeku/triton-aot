
import triton

@triton.jit
def mul(x, y):
    return x * y
