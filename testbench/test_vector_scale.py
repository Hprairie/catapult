import pytest
import torch
import catapult

@catapult.jit(kernel_path="cu/vector_scale.cu", kernel_name="vector_scale_kernel")
def vector_scale_kernel(blk, N, d_out, d_in, scalar, size):
    threads_per_block = min(size, 1024)
    blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

    print(f"Launching Vector Scale kernel: grid=({blocks_per_grid},1,1), block=({threads_per_block},1,1)")
    vector_scale_kernel.kernel[(blocks_per_grid, 1, 1), (threads_per_block, 1, 1)](d_out, d_in, scalar, size)

@pytest.mark.parametrize("N", [16, 32, 64, 128, 256, 512, 1024])
def test_vector_scale(cuda_device, N):
    scalar = 2.5
    inp = torch.rand(N, dtype=torch.float32, device=cuda_device)
    out = torch.empty(N, dtype=torch.float32, device=cuda_device)

    vector_scale_kernel(1, N, out, inp, scalar, N)

    ref = inp * scalar
    assert torch.allclose(out, ref, atol=1e-3), f"Vector scaling mismatch for N={N}"

