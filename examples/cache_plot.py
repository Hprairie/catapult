import pytest
import torch
import time
import numpy as np
import catapult

import matplotlib.pyplot as plt


@catapult.jit(kernel_path="example_kernel.cu", kernel_name="saxpy")
def testing(a, x, y, N):
    output = torch.zeros_like(x)

    testing.kernel[(32, 1, 1), (128, 1, 1)](a, x, y, output, N)
    return output


NUM_THREADS = 128
NUM_BLOCKS = 32
N = NUM_THREADS * NUM_BLOCKS


def test_kernel_cache_performance():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA device not available")

    # Prepare input data
    a = 2.0
    x = torch.rand(N, device=device, dtype=torch.float32)
    y = torch.rand(N, device=device, dtype=torch.float32)

    # Lists to store timing data
    execution_times = []

    # Run kernel 100 times and measure execution time
    for i in range(100):
        start_time = time.perf_counter()
        out = testing(a, x, y, N)
        torch.cuda.synchronize()  # Wait for kernel to complete
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        execution_times.append(execution_time)

        # Verify correctness
        expected = a * x + y
        assert torch.allclose(out, expected), f"Iteration {i}: Output mismatch"

    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101), execution_times, "b-", label="Execution Time")
    plt.axvline(x=1, color="r", linestyle="--", label="First Call (Cold Start)")
    plt.xlabel("Call Number")
    plt.ylabel("Execution Time (ms)")
    plt.title("Kernel Execution Time Over Multiple Calls")
    plt.legend()
    plt.grid(True)

    # Add annotations
    plt.annotate(
        f"Cold start: {execution_times[0]:.2f}ms",
        xy=(1, execution_times[0]),
        xytext=(10, execution_times[0] + 0.1),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    avg_cached = np.mean(execution_times[1:])
    plt.axhline(y=avg_cached, color="g", linestyle="--", label=f"Avg Cached: {avg_cached:.2f}ms")

    # Save the plot
    plt.savefig("kernel_cache_performance.png")

    # Print statistics
    print(f"\nPerformance Statistics:")
    print(f"First call (cold start): {execution_times[0]:.2f}ms")
    print(f"Average cached calls: {avg_cached:.2f}ms")
    print(f"Min cached time: {min(execution_times[1:]):.2f}ms")
    print(f"Max cached time: {max(execution_times[1:]):.2f}ms")

    # Verify caching is working
    assert execution_times[0] > avg_cached, "Caching doesn't seem to be effective"


if __name__ == "__main__":
    test_kernel_cache_performance()
