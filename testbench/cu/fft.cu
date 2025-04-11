extern "C" {

    #define PI 3.14159265358979323846
    #define THREADS_PER_BLOCK 1024
    #define TYPE_REAL float
    
    struct Complex {
        TYPE_REAL real;
        TYPE_REAL imag;
    
        __device__ Complex() : real(0), imag(0) {}
        __device__ Complex(TYPE_REAL r, TYPE_REAL i) : real(r), imag(i) {}
    

        __device__ Complex operator+(const Complex& other) const {
            return Complex(real + other.real, imag + other.imag);
        }
    

        __device__ Complex operator-(const Complex& other) const {
            return Complex(real - other.real, imag - other.imag);
        }
    

        __device__ Complex operator*(const Complex& other) const {
            return Complex(real * other.real - imag * other.imag, 
                           real * other.imag + imag * other.real);
        }
    };
    
    __global__ void fft(TYPE_REAL* d_in, Complex* d_complex_in, int size) {
        int tid  = threadIdx.x;
        int base = blockIdx.x * blockDim.x;
    
        int revIndex = 0, tempTid = tid;
        for (int i = 1; i < size; i <<= 1) {
            revIndex <<= 1;
            revIndex |= (tempTid & 1);
            tempTid >>= 1;
        }
    

        Complex val(d_in[2 * (base + revIndex)], d_in[2 * (base + revIndex) + 1]);
        __syncthreads();
        d_complex_in[base + tid] = val;
        __syncthreads();
    

        for (int n = 2, s = 1; n <= size; n <<= 1, s <<= 1) {
            int idxInGroup = tid % n;
            int k = (idxInGroup < s) ? idxInGroup : idxInGroup - s;
            TYPE_REAL angle = -2.0 * PI * k / n;
            Complex twiddle(cosf(angle), sinf(angle));
    
            Complex newVal;
            if (idxInGroup < s) {
                newVal = d_complex_in[base + tid] + twiddle * d_complex_in[base + tid + s];
            } else {
                newVal = d_complex_in[base + tid - s] - twiddle * d_complex_in[base + tid];
            }
    
            __syncthreads();
            d_complex_in[base + tid] = newVal;
            __syncthreads();
        }
    
        d_in[2 * (base + tid)] = d_complex_in[base + tid].real;
        d_in[2 * (base + tid) + 1] = d_complex_in[base + tid].imag;
    }
    
    __global__ void invfft(TYPE_REAL* d_in, Complex* d_complex_in, int size) {
        int tid  = threadIdx.x;
        int base = blockIdx.x * blockDim.x;
    
        int revIndex = 0, tempTid = tid;
        for (int i = 1; i < size; i <<= 1) {
            revIndex <<= 1;
            revIndex |= (tempTid & 1);
            tempTid >>= 1;
        }
    

        Complex val(d_in[2 * (base + revIndex)], d_in[2 * (base + revIndex) + 1]);
        __syncthreads();
        d_complex_in[base + tid] = val;
        __syncthreads();
    

        for (int n = 2, s = 1; n <= size; n <<= 1, s <<= 1) {
            int idxInGroup = tid % n;
            int k = (idxInGroup < s) ? idxInGroup : idxInGroup - s;
            TYPE_REAL angle = 2.0 * PI * k / n;
            Complex twiddle(cosf(angle), sinf(angle));
    
            Complex newVal;
            if (idxInGroup < s) {
                newVal = d_complex_in[base + tid] + twiddle * d_complex_in[base + tid + s];
            } else {
                newVal = d_complex_in[base + tid - s] - twiddle * d_complex_in[base + tid];
            }
    
            __syncthreads();
            d_complex_in[base + tid] = newVal;
            __syncthreads();
        }
    

        d_in[2 * (base + tid)] = d_complex_in[base + tid].real / size;
        d_in[2 * (base + tid) + 1] = d_complex_in[base + tid].imag / size;
    }
    
    } /* extern "C" */
    