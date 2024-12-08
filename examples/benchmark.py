import torch
import catapult
import time
import matplotlib.pyplot as plt
import saxpy_cuda
import numpy as np

NUM_THREADS = 128
NUM_BLOCKS = 32
N = NUM_THREADS * NUM_BLOCKS
NUM_RUNS = 100

@catapult.jit(
    kernel_path="example_kernel.cu",
    kernel_name="saxpy"
)
def catapult_saxpy(a, x, y):
    output = torch.zeros_like(x)
    catapult_saxpy.kernel[(NUM_BLOCKS, 1, 1), (NUM_THREADS, 1, 1)](a, x, y, output, N)
    return output

def benchmark():
    device = torch.device("cuda")
    a = 2.0
    x = torch.rand(N, device=device, dtype=torch.float32)
    y = torch.rand(N, device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        _ = catapult_saxpy(a, x, y)
        _ = saxpy_cuda.saxpy(a, x, y)
    
    torch.cuda.synchronize()
    
    catapult_times = []
    torch_ext_times = []
    
    # Benchmark runs
    for i in range(NUM_RUNS):
        # Catapult timing
        start = time.perf_counter()
        out_catapult = catapult_saxpy(a, x, y)
        torch.cuda.synchronize()
        catapult_times.append((time.perf_counter() - start) * 1000)
        
        # PyTorch Extension timing
        start = time.perf_counter()
        out_torch = saxpy_cuda.saxpy(a, x, y)
        torch.cuda.synchronize()
        torch_ext_times.append((time.perf_counter() - start) * 1000)
        
        # Verify results match
        if i == 0:
            assert torch.allclose(out_catapult, out_torch, rtol=1e-5, atol=1e-5)
    
    return catapult_times, torch_ext_times

def plot_results(catapult_times, torch_ext_times):
    plt.figure(figsize=(12, 6))
    x = range(len(catapult_times))
    
    plt.plot(x, catapult_times, 'b-', label='Catapult JIT', alpha=0.7)
    plt.plot(x, torch_ext_times, 'r-', label='PyTorch Extension', alpha=0.7)
    
    plt.xlabel('Run Number')
    plt.ylabel('Execution Time (ms)')
    plt.title('SAXPY Kernel Performance: Catapult vs PyTorch Extension')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics as text
    stats_text = (
        f'Catapult Mean: {np.mean(catapult_times):.3f}ms ± {np.std(catapult_times):.3f}ms\n'
        f'Torch Ext Mean: {np.mean(torch_ext_times):.3f}ms ± {np.std(torch_ext_times):.3f}ms'
    )
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
        
    print("Running benchmarks...")
    catapult_times, torch_ext_times = benchmark()
    
    print("Plotting results...")
    plot_results(catapult_times, torch_ext_times)
    
    print(f"Results saved to benchmark_results.png")
    print(f"Catapult average: {np.mean(catapult_times):.3f}ms")
    print(f"PyTorch Extension average: {np.mean(torch_ext_times):.3f}ms")