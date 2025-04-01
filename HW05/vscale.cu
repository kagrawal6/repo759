#include "vscale.cuh"

__global__ void vscale(const float* a, float* b, unsigned int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        b[i] = a[i] * b[i];
    }
}
