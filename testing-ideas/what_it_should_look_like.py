from numpy import zeros


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_S": 128, "BLOCK_T": 256}, num_warps=4),
        triton.Config(kwargs={"BLOCK_S": 128, "BLOCK_T": 256}, num_warps=4),
        triton.Config(kwargs={"BLOCK_S": 128, "BLOCK_T": 256}, num_warps=4),
        triton.Config(kwargs={"BLOCK_S": 128, "BLOCK_T": 256}, num_warps=4),
        triton.Config(kwargs={"BLOCK_S": 128, "BLOCK_T": 256}, num_warps=4),
    ],
    key=["n_elements"],  # the two above configs will be evaluated anytime
    # the value of x_size changes
)
@triton.heuristics(values={"BLOCK_T": lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
@triton.jit("path/to/file", "kernel_name", TEMPLATE_PARAMS=["BLOCK_T", "BLOCK_S"])
def add(kernel, x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    kernel[grid](x, y, output, n_elements)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


x, y, z, w

out = add(x, y)
out2 = add(z, w)
