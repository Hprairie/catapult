import torch
import catapult
import time
import matplotlib.pyplot as plt
import numpy as np
import matrix_cuda  # Import our custom CUDA module

MATRIX_SIZE = 1024
BLOCK_SIZE = 32
NUM_RUNS = 100

@catapult.jit(
    kernel_path="matrix.cu",
    kernel_name="matrix_ops"
)
def catapult_matmul(A, B, C, alpha, beta):
    output = torch.zeros_like(A)
    grid_dim = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE, 1)
    block_dim = (BLOCK_SIZE, BLOCK_SIZE, 1)
    catapult_matmul.kernel[grid_dim, block_dim](A, B, C, alpha, beta, output, MATRIX_SIZE)
    return output

def benchmark():
    device = torch.device("cuda")
    alpha = 2.0
    beta = 0.5
    
    A = torch.rand((MATRIX_SIZE, MATRIX_SIZE), device=device, dtype=torch.float32)
    B = torch.rand((MATRIX_SIZE, MATRIX_SIZE), device=device, dtype=torch.float32)
    C = torch.rand((MATRIX_SIZE, MATRIX_SIZE), device=device, dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        _ = catapult_matmul(A, B, C, alpha, beta)
        _ = matrix_cuda.matrix_ops(A, B, C, alpha, beta)
    
    torch.cuda.synchronize()
    
    catapult_times = []
    cuda_times = []
    
    for i in range(NUM_RUNS):
        start = time.perf_counter()
        out_catapult = catapult_matmul(A, B, C, alpha, beta)
        torch.cuda.synchronize()
        catapult_times.append((time.perf_counter() - start) * 1000)
        
        start = time.perf_counter()
        out_cuda = matrix_cuda.matrix_ops(A, B, C, alpha, beta)
        torch.cuda.synchronize()
        cuda_times.append((time.perf_counter() - start) * 1000)
        
        if i == 0:
            assert torch.allclose(out_catapult, out_cuda, rtol=1e-3, atol=1e-3)
    
    return catapult_times, cuda_times

def plot_results(catapult_times, cuda_times):
    plt.figure(figsize=(12, 6))
    x = range(len(catapult_times))
    
    plt.plot(x, catapult_times, 'b-', label='Catapult JIT', alpha=0.7)
    plt.plot(x, cuda_times, 'g-', label='Custom CUDA', alpha=0.7)
    
    plt.xlabel('Run Number')
    plt.ylabel('Execution Time (ms)')
    plt.title('Matrix Operations Performance Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    stats_text = (
        f'Catapult Mean: {np.mean(catapult_times):.3f}ms ± {np.std(catapult_times):.3f}ms\n'
        f'CUDA Mean: {np.mean(cuda_times):.3f}ms ± {np.std(cuda_times):.3f}ms'
    )
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('matmul_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
        
    print("Running matrix multiplication benchmarks...")
    catapult_times, cuda_times = benchmark()
    
    print("Plotting results...")
    plot_results(catapult_times, cuda_times)
    
    print(f"Results saved to matmul_benchmark_results.png")
    print(f"Catapult average: {np.mean(catapult_times):.3f}ms")
    print(f"Custom CUDA average: {np.mean(cuda_times):.3f}ms")