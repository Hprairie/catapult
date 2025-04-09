import torch
import torch.nn.functional as F
import triton
import catapult.library as lib

DEVICE = torch.device("cuda")


def unfused_attention(Qg, Kg, Vg):
    attention_matrix = torch.softmax(torch.einsum("b h i d, b h j d -> b h i j", Qg, Kg), dim=-1)
    return torch.einsum("b h i j, b h j d -> b h i d", attention_matrix, Vg)


fused_attention = lib.functions["attention"]

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['L'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['catapult', 'torch'],  # possible values for `line_arg``
        line_names=[
            "Catapult",
            "Torch",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="ms",  # label name for the y-axis
        plot_name="attention-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'B': 1, 'H': 1, "D": 128},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(B, L, H, D, provider):

    Qg = torch.ones(B, L, H, D, device=DEVICE, dtype=torch.bfloat16)
    Kg = torch.ones(B, L, H, D, device=DEVICE, dtype=torch.bfloat16)
    Vg = torch.ones(B, L, H, D, device=DEVICE, dtype=torch.bfloat16)

    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: unfused_attention(Qg, Kg, Vg), return_mode='median', warmup=100, rep=500)
    if provider == 'catapult':
        ms = triton.testing.do_bench(lambda: fused_attention(Qg, Kg, Vg, D=D), return_mode='median', warmup=100, rep=500)
    return ms


benchmark.run(show_plots=True, print_data=True)
