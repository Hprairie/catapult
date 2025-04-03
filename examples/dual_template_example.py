import torch
import catapult


@catapult.jit(
    kernel_path="example_templated_tk_kernel.cu",
    kernel_name="copy_kernel",
    kernel_param="globals",
    template_kernel=["PrintDebug"],  # Template the kernel name
    template_params=["CustomType"]    # Template the kernel parameter
)
def dual_template_demo(x):
    """
    Example showing both kernel name and parameter templating.
    
    This demonstrates using:
    - template_kernel to specialize the kernel function name
    - template_params to specialize the kernel parameter struct
    """
    output = torch.zeros_like(x)

    # Pass values for both template parameters
    dual_template_demo.kernel[(32, 1, 1), (128, 1, 1)](x, output, 
                                                      PrintDebug=True,  # Used for kernel name templating
                                                      CustomType=42)    # Used for parameter templating
    return output


# Allocate device memory using PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    raise RuntimeError("CUDA device not available.")
x = torch.ones(16, 32, 32, 64, device=device, dtype=torch.float32)

out = dual_template_demo(x)
print(f"Output shape: {out.shape}")
print(f"Output sum: {out.sum().item()}")