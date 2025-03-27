#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include "tuple"
using namespace kittens;

using my_layout = gl<float, -1, -1, -1, 64, st_fl<64,64>>;
struct globals {
    my_layout in, out;
    dim3 grid()  { return dim3(in.batch(), in.depth(), in.rows()); }
    dim3 block() { return dim3(in.cols()); }
    static auto inputs() {
        return std::make_tuple(&globals::in, &globals::out);
    }
};
__global__ void copy_kernel(const __grid_constant__ globals g) {
    if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) printf("Hello, from inside the kernel!\n");
    g.out[{blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x}] = g.in[{blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x}];
}