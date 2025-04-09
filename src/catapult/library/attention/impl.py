import torch
import catapult

@catapult.jit(
    kernel_path="attention.cuh",
    kernel_name="attend_ker",
    kernel_param="globals",
    template_kernel=["D"],
    template_params=["D"],
)
def attend(Qg, Kg, Vg, D):
    assert Qg.is_cuda and Kg.is_cuda and Vg.is_cuda, "All input tensors must be on a CUDA device"
    
    assert Qg.dim() == 4 and Kg.dim() == 4 and Vg.dim() == 4, \
        f"All input tensors must be 4-dimensional (B,N,H,D), got shapes: Qg={Qg.shape}, Kg={Kg.shape}, Vg={Vg.shape}"
    
    B, N, H, Dshape = Qg.shape
    
    assert Kg.shape[0] == B and Vg.shape[0] == B, \
        f"Batch size mismatch: Q={B}, K={Kg.shape[0]}, V={Vg.shape[0]}"
    
    assert Kg.shape[1] == N and Vg.shape[1] == N, \
        f"Sequence length mismatch: Q={N}, K={Kg.shape[1]}, V={Vg.shape[1]}"
    
    assert Kg.shape[2] == H and Vg.shape[2] == H, \
        f"Head count mismatch: Q={H}, K={Kg.shape[2]}, V={Vg.shape[2]}"
    
    assert Dshape == D and Kg.shape[3] == D and Vg.shape[3] == D, \
        f"Feature dimension mismatch: Expected {D}, got Q={Dshape}, K={Kg.shape[3]}, V={Vg.shape[3]}"
    
    Qg = Qg.contiguous()
    Kg = Kg.contiguous()
    Vg = Vg.contiguous()
    
    assert Qg.dtype == torch.bfloat16 and Kg.dtype == torch.bfloat16 and Vg.dtype == torch.bfloat16, \
        f"All tensors should be of dtype bfloat16, got: Q={Qg.dtype}, K={Kg.dtype}, V={Vg.dtype}"
    
    Og = torch.empty_like(Qg)
    
    attend.kernel(Qg, Kg, Vg, Og, D=D)
    return Og

def unfused_attend(Qg, Kg, Vg):
    attention_matrix = torch.softmax(torch.einsum("b h i d, b h j d -> b h i j", Qg, Kg), dim=-1)
    return torch.einsum("b h i j, b h j d -> b h i d", attention_matrix, Vg)


if __name__ == "__main__":
    device = torch.device("cuda")
    B = 1
    seq = 128 # ATTN_N
    H = 16  # ATTN_H
    D = 64

    Qg = torch.ones(B, seq, H, D, device=device, dtype=torch.bfloat16)
    Kg = torch.ones(B, seq, H, D, device=device, dtype=torch.bfloat16)
    Vg = torch.ones(B, seq, H, D, device=device, dtype=torch.bfloat16)

    Og = attend(Qg, Kg, Vg, D=D)
    print("Output tensor Og:")
    print(Og)
    Og_ref = unfused_attend(Qg, Kg, Vg)
    print("Output tensor Og_ref:")
    print(Og_ref)

func = attend