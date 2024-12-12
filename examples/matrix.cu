extern "C" __global__ void matrix_ops(
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