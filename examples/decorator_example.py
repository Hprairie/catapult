import torch
from catapult.runtime.jit import jit


@jit(kernel_path="example_kernel.cu", kernel_name="saxpy")
def testing(kernel, a, x, y, N):
    output = torch.zeros_like(x)

    kernel[(32, 1, 1), (128, 1, 1)](a, x, y, output, N)
    return output


NUM_THREADS = 128
NUM_BLOCKS = 32
N = NUM_THREADS * NUM_BLOCKS

# Allocate device memory using PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA device not available.")
a = 2.0
x = torch.rand(N, device=device, dtype=torch.float32)
y = torch.rand(N, device=device, dtype=torch.float32)

output = testing(a, x, y, N)
