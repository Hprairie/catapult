extern "C" {
    __global__ void vector_scale_kernel(float *d_out, float *d_in, float scalar, int size)
    {
        int myId = threadIdx.x + blockDim.x * blockIdx.x;

        if (myId < size) // Prevent out-of-bounds memory access
        {
            d_out[myId] = d_in[myId] * scalar;
        }
    }
}