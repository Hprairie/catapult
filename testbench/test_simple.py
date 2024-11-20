import torch
import catapult

@catapult.jit(kernel_path='cu/simple.cu',kernel_name='global_reduce_kernel')
def greduce(kernel, blk, N, din):
    output = torch.zeros_like(din)
    kernel[(blk, 1, 1), (N, 1, 1)](output, din)
    return output


def test_reduce(N):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device not available.")
    
    inp = torch.rand(N, device=device, dtype=torch.float32)
    exp = inp.sum()

    out = greduce(1, N, inp)[0]
    print(f'g: expect {exp}, get {out}')
    assert(torch.allclose(exp, out))
    


if __name__ == '__main__':
    n = 1
    while n <= 1024:
        test_reduce(n)
        n <<= 1
    print('PASS')
