extern "C" {
    // Forward pass kernel for vector addition
    __global__ void vectorAddKernel(float* a, float* b, float* c, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            c[tid] = a[tid] + b[tid];
        }
    }

    // Backward pass kernel for vector addition (gradient computation)
    __global__ void vectorAddBackwardKernel(float* grad_output, float* grad_a, float* grad_b, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            grad_a[tid] = grad_output[tid];
            grad_b[tid] = grad_output[tid];
        }
    }
}
