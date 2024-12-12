#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void matrix_ops_kernel(
    const float* A,
    const float* B,
    const float* C,
    float alpha,
    float beta,
    float* output,
    int N
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        
        // Matrix multiplication
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        // Complex operations
        float val = alpha * sum + beta * C[row * N + col];
        val = sinf(val) * cosf(val);
        val = fmaxf(val, 0.0f);
        val = tanhf(val);
        
        output[row * N + col] = val;
    }
}

torch::Tensor matrix_ops_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    float alpha,
    float beta
) {
    const int N = A.size(0);
    auto output = torch::zeros_like(A);
    
    const dim3 threads(32, 32);
    const dim3 blocks((N + threads.x - 1) / threads.x, 
                     (N + threads.y - 1) / threads.y);
    
    matrix_ops_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        alpha,
        beta,
        output.data_ptr<float>(),
        N
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_ops", &matrix_ops_cuda, "Complex Matrix Operations CUDA");
}