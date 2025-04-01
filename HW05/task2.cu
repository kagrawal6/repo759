// task2.cu
#include <cstdio>
#include <cuda.h>
#include <random>

// Kernel that computes a*x + y, where
//   x = threadIdx.x
//   y = blockIdx.x
//   a = an integer passed in from the host
// The result is stored in the dA array at a unique index per thread.
__global__ void computeKernel(int a, int* dA) {
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int x = threadIdx.x;  // "x"
    int y = blockIdx.x;   // "y"

    dA[globalId] = a * x + y;
}

int main() {
    const int N = 16;  // 16 total elements
    const int nBlocks = 2;
    const int nThreads = 8;
    int *dA = nullptr;
    int hA[N];

    // 1. Allocate array of 16 ints on the device
    cudaMalloc(&dA, N * sizeof(int));
    cudaMemset(dA, 0, N * sizeof(int));

    // 2. Generate a random integer 'a'
    //    (Here we generate an integer between 1 and 20, just as an example.)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 20);
    int a = dist(gen);

    // 3. Launch kernel with 2 blocks, 8 threads each
    computeKernel<<<nBlocks, nThreads>>>(a, dA);

    // 4. Copy results from device to host
    cudaMemcpy(hA, dA, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 5. Print the 16 values stored in hA
    //    All separated by a space, followed by a newline.
    for (int i = 0; i < N; i++) {
        printf("%d", hA[i]);
        if (i < N - 1) {
            printf(" ");
        } else {
            printf("\n");
        }
    }

    // 6. Cleanup
    cudaFree(dA);

    return 0;
}
