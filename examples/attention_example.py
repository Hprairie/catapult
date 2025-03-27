import torch
import catapult

# We assume D = 64 (NUM_WORKERS = 4 as in the CUDA globals)
D = 64


@catapult.jit(
    kernel_path="attention_tk.cuh",
    kernel_name="attend_ker",
    kernel_param="globals",
    template_params=["D"],
)
def attend(Qg, Kg, Vg):
    Og = torch.empty_like(Qg)
    attend.kernel(Qg, Kg, Vg, Og, D=D)
    return Og


if __name__ == "__main__":
    device = torch.device("cuda")
    # Define dimensions: (batch, sequence length, heads, feature dimension)
    B = 1
    seq = 768  # ATTN_N
    H = 16  # ATTN_H
    D = 64

    # Create dummy bf16 tensors for Q, K, and V.
    Qg = torch.ones(B, seq, H, D, device=device, dtype=torch.bfloat16)
    Kg = torch.ones(B, seq, H, D, device=device, dtype=torch.bfloat16)
    Vg = torch.ones(B, seq, H, D, device=device, dtype=torch.bfloat16)

    Og = attend(Qg, Kg, Vg)
    print("Output tensor Og:")
    print(Og)
