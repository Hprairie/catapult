#include <torch/extension.h>

__global__ void saxpy_kernel(float a, float *x, float *y, float *out, size_t n) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = a * x[tid] + y[tid];
    }
}

torch::Tensor saxpy_cuda(float a, torch::Tensor x, torch::Tensor y) {
    auto output = torch::zeros_like(x);
    
    const int threads = 128;
    const int blocks = (x.size(0) + threads - 1) / threads;
    
    saxpy_kernel<<<blocks, threads>>>(
        a,
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        x.size(0)
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("saxpy", &saxpy_cuda, "SAXPY CUDA");
}