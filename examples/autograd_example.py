import catapult
import torch

NUM_THREADS = 128
NUM_BLOCKS = 32
N = NUM_THREADS * NUM_BLOCKS


def setup_context(ctx, inputs, output):
    x, y = inputs
    ctx.x = x
    ctx.y = y


def register_fake(x, y):
    return torch.empty_like(x)


@catapult.jit(kernel_path="example_torch.cu", kernel_name="vectorAddBackwardKernel")
def add_bwd(kernel, ctx, grad_input):
    x_grad = torch.zeros_like(grad_input)
    y_grad = torch.zeros_like(grad_input)

    kernel[(NUM_BLOCKS, 1, 1), (NUM_THREADS, 1, 1)](grad_input, x_grad, y_grad, N)
    return x_grad, y_grad


@catapult.jit(kernel_path="example_torch.cu", kernel_name="vectorAddKernel")
def add_fwd(kernel, x, y):
    output = torch.zeros_like(x)

    kernel[(NUM_BLOCKS, 1, 1), (NUM_THREADS, 1, 1)](x, y, output, N)
    return output


# Create a wrapper function with explicit signature for custom_op
# Need to specify the types
@catapult.custom_op(
    name="mylib::add",
    mutates_args=(),
    device_types="cuda",
    backward_fn=add_bwd,
    setup_context=setup_context,
    register_fake=register_fake,
)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return add_fwd(x, y)


# Allocate device memory using PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA device not available.")
x = torch.rand(N, device=device, dtype=torch.float32, requires_grad=True)
y = torch.rand(N, device=device, dtype=torch.float32, requires_grad=True)
x_temp = x.detach().clone().requires_grad_()
y_temp = y.detach().clone().requires_grad_()

torch.library.opcheck(add, (x, y), test_utils="test_aot_dispatch_static")

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
