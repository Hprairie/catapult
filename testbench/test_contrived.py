import pytest
import torch
import catapult
import time
import matplotlib.pyplot as plt
import os

# will be really bad
def toy_layer_pytorch(X):
    y = X.clone()
    for _ in range(20):
        y = y * 1.00001 + 0.00001 * torch.sin(y)
    return y

@catapult.jit(kernel_path="cu/contrived.cu", kernel_name="toy_layer_kernel")
def toy_layer_kernel(blk, B, D, X, Y):
    threads_per_block = 256
    total = B * D
    blocks_per_grid = (total + threads_per_block - 1) // threads_per_block
    grid = (blocks_per_grid, 1, 1)
    block = (threads_per_block, 1, 1)

    print(f"Launching Toy Layer kernel: grid={grid}, block={block}")
    toy_layer_kernel.kernel[grid, block](X, Y, B, D)

def benchmark_toy_layer():
    sizes = [128, 256, 512, 1024, 2048, 4096]
    pytorch_times = []
    cuda_times = []

    for B in sizes:
        D = 128
        X = torch.rand(B, D, device="cuda", dtype=torch.float32)
        Y_ref = torch.empty_like(X)
        Y_cuda = torch.empty_like(X)

        torch.cuda.synchronize()
        t0 = time.time()
        Y_ref = toy_layer_pytorch(X)
        torch.cuda.synchronize()
        pytorch_times.append(time.time() - t0)

        torch.cuda.synchronize()
        t0 = time.time()
        toy_layer_kernel(1, B, D, X, Y_cuda)
        torch.cuda.synchronize()
        cuda_times.append(time.time() - t0)

        assert torch.allclose(Y_ref, Y_cuda, atol=1e-4), f"Mismatch at B={B}"

    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(sizes, pytorch_times, label="PyTorch")
    plt.plot(sizes, cuda_times, label="Fused CUDA Kernel")
    plt.xlabel("Batch size (B)")
    plt.ylabel("Time (s)")
    plt.title("Toy Layer Performance (Including First Run)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/toy_layer_full.png")

    plt.figure()
    plt.plot(sizes[1:], pytorch_times[1:], label="PyTorch")
    plt.plot(sizes[1:], cuda_times[1:], label="Fused CUDA Kernel")
    plt.xlabel("Batch size (B)")
    plt.ylabel("Time (s)")
    plt.title("Toy Layer Performance (Warm Runs Only)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/toy_layer_trimmed.png")

    print("saved to results/toy_layer_full.png and toy_layer_trimmed.png")

    avg_pytorch = sum(pytorch_times[1:]) / len(pytorch_times[1:])
    avg_cuda = sum(cuda_times[1:]) / len(cuda_times[1:])
    speedup = avg_pytorch / avg_cuda

    print(f"\n[warm Run Averages]")
    print(f"pyTorch: {avg_pytorch * 1000:.3f} ms")
    print(f"catapult: {avg_cuda * 1000:.3f} ms")
    print(f"speedup: {speedup:.2f}x faster")


if __name__ == "__main__":
    benchmark_toy_layer()
