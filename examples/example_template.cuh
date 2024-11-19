template<size_t N>
__global__ void saxpy(float a, float* x, float* y, float* out)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        out[tid] = a * x[tid] + y[tid];
    }
}
