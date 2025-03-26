import pytest
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import catapult

@catapult.jit(kernel_path="cu/fft.cu", kernel_name="fft")
def fft_kernel(blk, N, din, dcin, size):
    threads_per_block = min(size, 1024)
    blocks_per_grid = 1

    print(f"Launching FFT kernel: grid=({blocks_per_grid},1,1), block=({threads_per_block},1,1)")

    fft_kernel.kernel[(blocks_per_grid, 1, 1), (threads_per_block, 1, 1)](din, dcin, size)

@catapult.jit(kernel_path="cu/fft.cu", kernel_name="invfft")
def ifft_kernel(blk, N, din, dcin, size):
    threads_per_block = min(size, 1024)
    blocks_per_grid = 1

    print(f"Launching IFFT kernel: grid=({blocks_per_grid},1,1), block=({threads_per_block},1,1)")

    ifft_kernel.kernel[(blocks_per_grid, 1, 1), (threads_per_block, 1, 1)](din, dcin, size)

@pytest.mark.parametrize("N", [16, 32, 64, 128, 256, 512, 1024])
def test_fft_correctness(cuda_device, N):
    """Test fft kernel correctness against torch's fft implementation."""
    if (N & (N - 1)) != 0:
        raise ValueError(f"N must be a power of two! Got N={N}")

    real = torch.rand(N, dtype=torch.float32, device=cuda_device)
    imag = torch.rand(N, dtype=torch.float32, device=cuda_device)

    inp = torch.empty(2 * N, dtype=torch.float32, device=cuda_device)
    inp[0::2] = real
    inp[1::2] = imag

    dcin = torch.empty(2 * N, dtype=torch.float32, device=cuda_device)  # Intermediate complex buffer

    fft_kernel(1, N, inp, dcin, N)

    out_complex = torch.view_as_complex(dcin.view(-1, 2).contiguous())

    ref_fft = torch.fft.fft(real + 1j * imag)

    assert torch.allclose(out_complex, ref_fft, atol=1e-3), f"FFT mismatch for N={N}"

@pytest.mark.parametrize("N", [16, 32, 64, 128, 256, 512, 1024])
def test_fft_inverse(cuda_device, N):
    if (N & (N - 1)) != 0:
        raise ValueError(f"N must be a power of two! Got N={N}")

    real = torch.rand(N, dtype=torch.float32, device=cuda_device)
    imag = torch.rand(N, dtype=torch.float32, device=cuda_device)

    inp = torch.empty(2 * N, dtype=torch.float32, device=cuda_device)
    inp[0::2] = real
    inp[1::2] = imag

    dcin = torch.empty(2 * N, dtype=torch.float32, device=cuda_device)  # Intermediate complex buffer

    fft_kernel(1, N, inp, dcin, N)
    ifft_kernel(1, N, dcin, inp, N)

    assert torch.allclose(inp, inp, atol=1e-3), f"Inverse FFT failed for N={N}"

def benchmark_fft(cuda_device):
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    times = []

    for N in sizes:
        real = torch.rand(N, dtype=torch.float32, device=cuda_device)
        imag = torch.rand(N, dtype=torch.float32, device=cuda_device)

        inp = torch.empty(2 * N, dtype=torch.float32, device=cuda_device)
        inp[0::2] = real
        inp[1::2] = imag
        dcin = torch.empty(2 * N, dtype=torch.float32, device=cuda_device)  # Intermediate complex buffer

        print(f"Benchmarking FFT kernel: blk=(1,1,1), thread=({N},1,1)")

        start = time.time()
        for _ in range(10):
            fft_kernel(1, N, inp, dcin, N)
            torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) / 10
        times.append(avg_time)

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, times, marker="o", linestyle="-", label="FFT Execution Time")
    plt.xlabel("Input Size (N)")
    plt.ylabel("Time (seconds)")
    plt.title("FFT Performance Benchmark")
    plt.legend()
    plt.grid()
    plt.savefig("fft_benchmark.png")
    print("Benchmark graph saved as fft_benchmark.png")

if __name__ == "__main__":
    benchmark_fft(torch.device("cuda"))
