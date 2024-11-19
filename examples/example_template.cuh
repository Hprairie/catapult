template<size_t N>
__global__ void saxpy(float a, const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        out[tid] = a * x[tid] + y[tid];
    }
}
