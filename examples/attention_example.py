import torch
import catapult

# We assume D = 64 (NUM_WORKERS = 4 as in the CUDA globals)
D = 64


@catapult.jit(
    kernel_path="attention_tk.cuh",
    kernel_name="attend_ker",
    kernel_param="globals",
    template_kernel=["D"],
    template_params=["D"],
)
def attend(Qg, Kg, Vg):
    Og = torch.empty_like(Qg)
    attend.kernel(Qg, Kg, Vg, Og, D=D)
    return Og

def unfused_attend(Qg, Kg, Vg):
    attention_matrix = torch.softmax(torch.einsum("b h i d, b h j d -> b h i j", Qg, Kg), dim=-1)
    # mask = torch.triu(torch.ones_like(attention_matrix), diagonal=1).bool()
    # attention_matrix = attention_matrix.masked_fill(mask, 0)
    return torch.einsum("b h i j, b h j d -> b h i d", attention_matrix, Vg)


if __name__ == "__main__":
    device = torch.device("cuda")
    # Define dimensions: (batch, sequence length, heads, feature dimension)
    B = 1
    seq = 128 # ATTN_N
    H = 16  # ATTN_H
    D = 64

    # Create dummy bf16 tensors for Q, K, and V.
    Qg = torch.ones(B, seq, H, D, device=device, dtype=torch.bfloat16)
    Kg = torch.ones(B, seq, H, D, device=device, dtype=torch.bfloat16)
    Vg = torch.ones(B, seq, H, D, device=device, dtype=torch.bfloat16)

    Og = attend(Qg, Kg, Vg)
    print("Output tensor Og:")
    print(Og)
    Og_ref = unfused_attend(Qg, Kg, Vg)
    print("Output tensor Og_ref:")
    print(Og_ref)
