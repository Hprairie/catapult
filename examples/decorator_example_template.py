import torch
import catapult


@catapult.autotune(
    configs=[
        catapult.Config({
            'NUM_THREADS': 128,
            'NUM_BLOCKS': 32,
        }),
        catapult.Config({
            'NUM_THREADS': 256,
            'NUM_BLOCKS': 16,
        }),
        catapult.Config({
            'NUM_THREADS': 64,
            'NUM_BLOCKS': 64,
        }),
    ],
    key=['N'],  # Retune when N changes
)
@catapult.jit(
    kernel_path="example_template.cuh",
    kernel_name="saxpy",
    template_params=["N"],
)
def testing(a, x, y, N, NUM_THREADS, NUM_BLOCKS):
    output = torch.zeros_like(x)
    # Grid/block config comes from the selected Config object
    testing.kernel[(NUM_BLOCKS, 1, 1), (NUM_THREADS, 1, 1)](a, x, y, output, N=N)
    return output


# Test with different sizes
N = 4096  # This matches our configs
device = torch.device("cuda")
a = 2.0
x = torch.rand(N, device=device, dtype=torch.float32)
y = torch.rand(N, device=device, dtype=torch.float32)

out = testing(a, x, y, N)
expected = a * x + y
print(out, expected)
assert torch.allclose(out, expected)

# Test with different N to trigger re-autotuning
N = 4096
x = torch.rand(N, device=device, dtype=torch.float32)
y = torch.rand(N, device=device, dtype=torch.float32)
out = testing(a, x, y, N)  # Will trigger new autotuning
