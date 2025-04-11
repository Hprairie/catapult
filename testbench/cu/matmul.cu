extern "C" {

    __global__ void matmul_kernel(float *A, float *B, float *C, int N)
    {
        __shared__ float shared_A[32][32];
        __shared__ float shared_B[32][32];
    
        int row = threadIdx.y + blockIdx.y * blockDim.y;
        int col = threadIdx.x + blockIdx.x * blockDim.x;
    
        float sum = 0.0f;
        for (int i = 0; i < N / 32; i++) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * N + i * 32 + threadIdx.x];
            shared_B[threadIdx.y][threadIdx.x] = B[(i * 32 + threadIdx.y) * N + col];
            __syncthreads();
    
            for (int j = 0; j < 32; j++) {
                sum += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
            }
            __syncthreads();
        }
    
        C[row * N + col] = sum;
    }
    
    }
