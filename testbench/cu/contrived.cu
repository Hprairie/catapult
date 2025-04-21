extern "C" {

    __global__ void toy_layer_kernel(float* X, float* Y, int B, int D) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = B * D;
        if (idx >= total) return;
    
        float x = X[idx];
        for (int i = 0; i < 20; i++) {
            x = x * 1.00001f + 0.00001f * sinf(x);
        }
        Y[idx] = x;
    }
    
    }
    