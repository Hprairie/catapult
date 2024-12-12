import torch
import numpy as np
from line_profiler import profile
import catapult

# Kernel configuration parameters
import torch
import numpy as np
import line_profiler
import catapult

# Kernel configuration parameters
NUM_THREADS = 128
NUM_BLOCKS = 32
N = NUM_THREADS * NUM_BLOCKS

@catapult.jit(
    kernel_path="example_kernel.cu",
    kernel_name="saxpy"
)
@profile
def catapult_saxpy(a, x, y):
    output = torch.zeros_like(x)
    catapult_saxpy.kernel[(NUM_BLOCKS, 1, 1), (NUM_THREADS, 1, 1)](a, x, y, output, N)
    return output

def main():
    # Generate input data
    a = 2.0  # Scalar multiplier
    x = torch.rand(N, dtype=torch.float32, device='cuda')
    y = torch.rand(N, dtype=torch.float32, device='cuda')

    # Warm-up run (JIT compilation)
    _ = catapult_saxpy(a, x, y)

    # Actual profiled run
    result = catapult_saxpy(a, x, y)

if __name__ == '__main__':
    # Run the main function
    main()

# Profiling instructions:
# 1. Save this script as saxpy_profile.py
# 2. Install line_profiler: pip install line_profiler
# 3. Run with: kernprof -l -v saxpy_profile.py