import pytest
import torch
import catapult

@catapult.jit(kernel_path="cu/matmul.cu", kernel_name="matmul_kernel")
def matmul_kernel(blk, N, A, B, C):
    threads_per_block = min(32, N)  # Typically, 32 is used for matrix multiplication
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    print(f"Launching Matrix Multiply kernel: grid=({blocks_per_grid}, {blocks_per_grid}, 1), block=({threads_per_block}, {threads_per_block}, 1)")

    matmul_kernel.kernel[(blocks_per_grid, blocks_per_grid, 1), (threads_per_block, threads_per_block, 1)](A, B, C, N)

@pytest.mark.parametrize("N", [32, 64, 128, 256, 512, 1024])
def test_matmul(cuda_device, N):
    A = torch.rand(N, N, dtype=torch.float32, device=cuda_device)
    B = torch.rand(N, N, dtype=torch.float32, device=cuda_device)
    C = torch.empty(N, N, dtype=torch.float32, device=cuda_device)

    matmul_kernel(1, N, A, B, C)

    ref = torch.matmul(A, B)
    assert torch.allclose(C, ref, atol=1e-3), f"Matrix multiplication mismatch for N={N}"
