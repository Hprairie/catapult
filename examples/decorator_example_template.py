import torch
import catapult
from cuda import nvrtc


@catapult.jit(
    kernel_path="example_template.cuh",
    kernel_name="saxpy",
    template_params=["N"],
    compile_options=["--std=c++14"],  # Added more options
)
def testing(kernel, a, x, y, N):
    try:
        output = torch.zeros_like(x)
        kernel[(32, 1, 1), (128, 1, 1)](a, x, y, output, N=N)
        return output
    except RuntimeError as e:
        # Get the compilation log
        program = kernel.kernel_params.program.program
        log_size = nvrtc.nvrtcGetProgramLogSize(program)[1]
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(program, log)
        print("Compilation Error Log:")
        print(log.decode())
        raise e


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

out = testing(a, x, y, N)
expected = a * x + y
print(out, expected)
if not torch.allclose(out, expected):
    raise ValueError("Kernel output does not match expected result")
