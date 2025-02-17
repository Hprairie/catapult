import catapult
import torch
from typing import Tuple

NUM_THREADS = 128
NUM_BLOCKS = 32
N = NUM_THREADS * NUM_BLOCKS



def register_fake_bwd(out):
    return torch.empty_like(out), torch.empty_like(out)

@catapult.custom_op("mylib::add_bwd", mutates_args=(), device_types="cuda", register_fake=register_fake_bwd)
@catapult.jit(kernel_path="example_torch.cu", kernel_name="vectorAddBackwardKernel")
def add_bwd(grad_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x_grad = torch.zeros_like(grad_input)
    y_grad = torch.zeros_like(grad_input)

    add_bwd.kernel[(NUM_BLOCKS, 1, 1), (NUM_THREADS, 1, 1)](grad_input, x_grad, y_grad, N)
    return x_grad, y_grad

def add_bwd_f(ctx, input_grad):
    return add_bwd(input_grad)

def setup_context(ctx, inputs, output):
    x, y = inputs
    ctx.x = x
    ctx.y = y

def register_fake(x, y):
    return torch.empty_like(x)


@catapult.custom_op("mylib::add", mutates_args=(), device_types="cuda", register_fake=register_fake, backward_fn=add_bwd_f, setup_context=setup_context)
@catapult.jit(kernel_path="example_torch.cu", kernel_name="vectorAddKernel")
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(x)

    add.kernel[(NUM_BLOCKS, 1, 1), (NUM_THREADS, 1, 1)](x, y, output, N)
    return output


@torch.compile(fullgraph=True)
def compiled_add(x, y):
    return add(x, y)


# Allocate device memory using PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA device not available.")
x = torch.rand(N, device=device, dtype=torch.float32, requires_grad=True)
y = torch.rand(N, device=device, dtype=torch.float32, requires_grad=True)
x_temp = x.detach().clone().requires_grad_()
y_temp = y.detach().clone().requires_grad_()

# torch.library.opcheck(add, (x, y), test_utils="test_aot_dispatch_static")

out = compiled_add(x, y)
print(out)

out = add(x, y)
expected = x_temp + y_temp
print(out, expected)
if not torch.allclose(out, expected):
    raise ValueError("Kernel output does not match expected result")

# Check Autograd functionality
grad = torch.rand_like(out)
out.backward(grad)
expected.backward(grad)
print(x.grad, x_temp.grad)
print(y.grad, y_temp.grad)
if not torch.allclose(x.grad, x_temp.grad):
    raise ValueError("Kernel output does not match expected result")
if not torch.allclose(y.grad, y_temp.grad):
    raise ValueError("Kernel output does not match expected result")
