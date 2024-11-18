<h1 align="center" style="fontsize:50em"><b>CATapult</b></h1>

![mascot](assets/mascot.jpg)


# Installing

Currently the project can be installed by cloning it down and then running the following in the root directory

```bash
pip install -e .
```


# Usage

Consider the following file `example_kernel.cu` which is defined below:

```cuda
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a * x[tid] + y[tid];
    }
}
```

To use catapult's auto-jit compiler we can create the following script with a defined kernel launching function.

```python
import torch
import catapult


@catapult.jit(kernel_path="example_kernel.cu", kernel_name="saxpy")
def testing(kernel, a, x, y, N):
    output = torch.zeros_like(x)

    kernel[(32, 1, 1), (128, 1, 1)](a, x, y, output, N)
    return output
```

We can then use the function like built-in pytorch functions.

```python
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
```
