import ctypes
import numpy as np
import torch
from cuda import cuda, nvrtc


def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


saxpy = """\
extern "C" __global__
void saxpy(float a, float *x, float *y, float *out, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a * x[tid] + y[tid];
    }
}
"""


def main():
    # Init CUDA
    torch.cuda.init()
    # (err,) = cuda.cuInit(0)
    # ASSERT_DRV(err)

    print(torch.cuda.current_device())

    stream = torch.cuda.current_stream()
    print(stream.cuda_stream)
    if stream.cuda_stream == 0:
        new_stream = torch.cuda.Stream()
        torch.cuda.set_stream(new_stream)
    stream = torch.cuda.current_stream()
    print(stream.cuda_stream)
    # Device and context
    # print(torch.cuda.current_stream())
    # err, context = cuda.cuCtxGetCurrent()
    # ASSERT_DRV(err)
    err, cuDevice = cuda.cuDeviceGet(torch.cuda.current_device())
    ASSERT_DRV(err)
    # print(context)

    # Create and compile program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(saxpy), b"saxpy.cu", 0, None, None)
    ASSERT_DRV(err)

    # Get compute capability and architecture argument
    err, major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice
    )
    ASSERT_DRV(err)
    err, minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice
    )
    ASSERT_DRV(err)
    arch_arg = bytes(f"--gpu-architecture=sm_{major}{minor}", "ascii")

    # Compile program
    opts = [b"--fmad=false", arch_arg]
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    ASSERT_DRV(err)

    # Get compiled data (CUBIN or PTX)
    err, dataSize = nvrtc.nvrtcGetPTXSize(prog)
    ASSERT_DRV(err)
    data = b" " * dataSize
    (err,) = nvrtc.nvrtcGetPTX(prog, data)
    ASSERT_DRV(err)

    # Load compiled data as module and get function
    data = np.char.array(data)
    err, module = cuda.cuModuleLoadData(data)
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(module, b"saxpy")
    ASSERT_DRV(err)

    print(stream)
    # err, stream = cuda.cuStreamCreate(0)
    # print(type(stream))
    # stream = cuda.CUstream(stream.cuda_stream)
    # # ASSERT_DRV(err)
    # print(stream)

    (err,) = cuda.cuStreamSynchronize(stream.cuda_stream)
    ASSERT_DRV(err)

    # Kernel configuration
    NUM_THREADS = 128
    NUM_BLOCKS = 32
    N = NUM_THREADS * NUM_BLOCKS

    # Allocate device memory using PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device not available.")

    a = torch.tensor(2.0, device=device, dtype=torch.float32)
    x = torch.rand(N, device=device, dtype=torch.float32)
    y = torch.rand(N, device=device, dtype=torch.float32)
    out = torch.zeros(N, device=device, dtype=torch.float32)

    # Define argument pointers for cuLaunchKernel
    arg_values = (ctypes.c_float(a.item()), x.data_ptr(), y.data_ptr(), out.data_ptr(), ctypes.c_size_t(N))
    arg_types = (ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)

    # Launch kernel
    (err,) = cuda.cuLaunchKernel(
        kernel,
        NUM_BLOCKS,
        1,
        1,  # grid dimensions
        NUM_THREADS,
        1,
        1,  # block dimensions
        0,
        stream.cuda_stream,  # shared memory and stream
        (arg_values, arg_types),
        0,  # arguments
    )
    ASSERT_DRV(err)

    # Synchronize
    (err,) = cuda.cuCtxSynchronize()
    ASSERT_DRV(err)

    # Validate results
    expected = a * x + y
    if not torch.allclose(out, expected):
        raise ValueError("Kernel output does not match expected result")

    # Cleanup
    (err,) = cuda.cuModuleUnload(module)
    ASSERT_DRV(err)
    # (err,) = cuda.cuCtxDestroy(context)
    # ASSERT_DRV(err)


if __name__ == "__main__":
    main()
