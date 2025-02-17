import pytest
import torch
import catapult

@catapult.jit(kernel_path='cu/simple.cu', kernel_name='global_reduce_kernel')
def greduce(kernel, blk, N, din):
    output = torch.zeros_like(din)
    kernel[(blk, 1, 1), (N, 1, 1)](output, din)
    return output

@pytest.fixture
def cuda_device_verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("Skipping test: CUDA device not available")
    return device

@pytest.mark.parametrize("N", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
def test_reduce(cuda_device, N):
""" test a basic reduce kernel on various input sizes """
    inp = torch.rand(N, device=cuda_device, dtype=torch.float32)
    exp = inp.sum()

    out = greduce(1, N, inp)[0]

    assert torch.allclose(exp, out), f"Expected {exp}, but got {out}"
