import torch
from line_profiler import profile
import catapult
import numpy as np

# Matrix dimensions
MATRIX_SIZE = 1024
BLOCK_SIZE = 32

@catapult.jit(
    kernel_path="matrix.cu",
    kernel_name="matrix_ops"
)
@profile
def catapult_matmul(A, B, C, alpha, beta):
    output = torch.zeros_like(A)
    grid_dim = (MATRIX_SIZE // BLOCK_SIZE, MATRIX_SIZE // BLOCK_SIZE, 1)
    block_dim = (BLOCK_SIZE, BLOCK_SIZE, 1)
    catapult_matmul.kernel[grid_dim, block_dim](A, B, C, alpha, beta, output, MATRIX_SIZE)
    return output

@profile
def torch_matmul(A, B, C, alpha, beta):
    return torch.matmul(A, B) * alpha + C * beta

def print_memory_stats():
    print("\nGPU Memory Statistics:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")

    device = torch.device("cuda")
    alpha = 2.0
    beta = 0.5

    # Generate input data
    A = torch.rand((MATRIX_SIZE, MATRIX_SIZE), device=device, dtype=torch.float32)
    B = torch.rand((MATRIX_SIZE, MATRIX_SIZE), device=device, dtype=torch.float32)
    C = torch.rand((MATRIX_SIZE, MATRIX_SIZE), device=device, dtype=torch.float32)

    print("Input Tensors:")
    print(f"A shape: {A.shape}, device: {A.device}")
    print(f"B shape: {B.shape}, device: {B.device}")
    print(f"C shape: {C.shape}, device: {C.device}")

    # Warmup runs
    print("\nPerforming warmup runs...")
    for _ in range(5):
        _ = catapult_matmul(A, B, C, alpha, beta)
        _ = torch_matmul(A, B, C, alpha, beta)
    torch.cuda.synchronize()

    print("\nRunning profiled operations...")
    
    # Profile Catapult implementation
    print("\nProfiling Catapult implementation:")
    result_catapult = catapult_matmul(A, B, C, alpha, beta)
    torch.cuda.synchronize()
    
    # Profile PyTorch implementation
    print("\nProfiling PyTorch implementation:")
    result_torch = torch_matmul(A, B, C, alpha, beta)
    torch.cuda.synchronize()

    # Verify results match
    print("\nVerifying results...")
    match = torch.allclose(result_catapult, result_torch, rtol=1e-3, atol=1e-3)
    print(f"Results match: {match}")
    if not match:
        max_diff = torch.max(torch.abs(result_catapult - result_torch))
        print(f"Maximum difference: {max_diff}")

    print_memory_stats()

if __name__ == '__main__':
    main()

# Profiling instructions:
# 1. Install line_profiler: pip install line_profiler
# 2. Run with: kernprof -l -v better_profile.py