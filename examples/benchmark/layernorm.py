import torch
import torch.nn.functional as F
import triton
import catapult.library as lib

DEVICE = torch.device("cuda")

# def unfused_layernorm(x, residual, norm_weight, norm_bias, B, N, D, dropout_p=0.0):
#     x = x + residual
#     o_resid = x
#     x_norm = x - torch.mean(x, dim=-1, keepdim=True)
#     x_var = torch.mean(x_norm ** 2, dim=-1, keepdim=True)
#     x_norm = x_norm / (torch.sqrt(x_var) + 1e-6)
#     x = x_norm * norm_weight + norm_bias
#     return x, o_resid

def unfused_layernorm(x, residual, norm_weight, norm_bias, B, N, D, dropout_p=0.0):
    x = x + residual
    o_resid = x
    F.layer_norm(x, (D,), norm_weight, norm_bias)
    return x, o_resid


fused_layernorm = lib.functions["layernorm"]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['D'],  # argument names to use as an x-axis for the plot
        x_vals=[32 * i for i in range(2, 30)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['catapult', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Catapult",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="layernorm-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'B': 32, 'N': 1024},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(B, N, D, provider):
    x = torch.ones(B, N, D, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.ones_like(x)
    norm_weight = torch.ones(D, device=DEVICE, dtype=torch.bfloat16)
    norm_bias = torch.zeros(D, device=DEVICE, dtype=torch.bfloat16)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: unfused_layernorm(x, residual, norm_weight, norm_bias, B, N, D), return_mode='median', warmup=100, rep=500)
    if provider == 'catapult':
        ms = triton.testing.do_bench(lambda: fused_layernorm(x, residual, norm_weight, norm_bias, B, N, D), return_mode='median', warmup=100, rep=500)
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return ms


benchmark.run(show_plots=True, print_data=True)
