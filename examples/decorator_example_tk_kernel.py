import torch
import catapult


@catapult.jit(
    kernel_path="example_tk_kernel.cu",
    kernel_name="copy_kernel",
    kernel_param="globals",
)
def testing(x):
    output = torch.zeros_like(x)

    b, c, h, w = x.shape

    testing.kernel[(b, c, h), (w, 1, 1)](x, output)
    return output


NUM_THREADS = 128
NUM_BLOCKS = 32
N = NUM_THREADS * NUM_BLOCKS

# Allocate device memory using PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA device not available.")
x = torch.ones(1, 1, 1, 64, device=device, dtype=torch.float32)

out = testing(x)
print(out)
