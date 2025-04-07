import torch
import catapult

@catapult.jit(
    kernel_path="rmsnorm_tk.cuh",
    kernel_name="rmsnorm_tk",
    kernel_param="norm_globals",
    template_kernel=["D", "V"],
    template_params=["D", "V"],
)
def fused_layernorm_tk(x, residual, norm_weight, norm_bias, B, N, D, dropout_p=0.0):
    assert x.dim() == 3, "Input x must be 3-dimensional (B,N,D)"
    assert residual.dim() == 3, "Input residual must be 3-dimensional (B,N,D)"
    assert norm_weight.dim() == 1, "Norm weight must be 1-dimensional (D)"
    assert norm_bias.dim() == 1, "Norm bias must be 1-dimensional (D)"
    
    # Check dimensions
    assert x.size(0) == B, f"Batch size mismatch: {x.size(0)} vs {B}"
    assert x.size(1) == N, f"Sequence length mismatch: {x.size(1)} vs {N}"
    assert x.size(2) == D, f"Model dimension mismatch: {x.size(2)} vs {D}"
    assert residual.size(0) == B, f"Residual batch mismatch: {residual.size(0)} vs {B}"
    assert residual.size(1) == N, f"Residual sequence length mismatch: {residual.size(1)} vs {N}"
    assert residual.size(2) == D, f"Residual dimension mismatch: {residual.size(2)} vs {D}"
    assert norm_weight.size(0) == D, f"Norm weight dimension mismatch: {norm_weight.size(0)} vs {D}"
    assert norm_bias.size(0) == D, f"Norm bias dimension mismatch: {norm_bias.size(0)} vs {D}"
    
    # Check divisibility for alignment - assuming TILE_ROW_DIM is 16 for bf16
    TILE_ROW_DIM = 16  # This should match kittens::TILE_ROW_DIM<bf16>
    assert N % TILE_ROW_DIM == 0, f"Sequence length ({N}) must be divisible by {TILE_ROW_DIM}"

    # Create output tensors
    o = torch.empty_like(x)
    o_resid = torch.empty_like(x)

    # Make sure they tensor are 4 dimensional
    x = x[:, None, :, :]
    residual = residual[:, None, :, :]
    norm_weight = norm_weight[None, None, None, :]
    norm_bias = norm_bias[None, None, None, :]
    o = o[:, None, :, :]
    o_resid = o_resid[:, None, :, :]
    
    # Launch the kernel
    fused_layernorm_tk.kernel(
        x, residual, o, o_resid, norm_weight, norm_bias, V=dropout_p, D=D
    )
    return o, o_resid

def unfused_rmsnorm(x, residual, norm_weight, norm_bias, B, N, D, dropout_p=0.0):
    # Torch implementation of RMSNorm
    x = x + residual
    o_resid = x
    norm = torch.mean(x ** 2, dim=-1, keepdim=True)
    x = x / torch.sqrt(norm + 1e-6)
    x = x * norm_weight + norm_bias
    return x, o_resid

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda")
    B, N, D = 8, 16, 1024
    x = torch.ones(B, N, D, device=device, dtype=torch.bfloat16)
    residual = torch.ones_like(x)
    norm_weight = torch.ones(D, device=device, dtype=torch.bfloat16)
    norm_bias = torch.zeros(D, device=device, dtype=torch.bfloat16)
    o, o_resid = fused_layernorm_tk(x, residual, norm_weight, norm_bias, B, N, D, dropout_p=0.0)
    print("Outputs:", o, o_resid)
    o_ref, o_resid_ref = unfused_rmsnorm(x, residual, norm_weight, norm_bias, B, N, D, dropout_p=0.0)
    print("Reference Outputs:", o_ref, o_resid_ref)

func = fused_layernorm_tk