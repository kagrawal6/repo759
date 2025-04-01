#include <cuda.h>
#include <stdio.h>

__global__ void factorialKernel() {
    int idx = threadIdx.x;  // thread index in this block
    int n = idx + 1;        // integer value for factorial

    int fact = 1;
    for (int i = 2; i <= n; i++) {
        fact *= i;
    }

    printf("%d != %d\n", n, fact);	
}

int main() {
    const int nThreads = 8;
	const int nBlocks = 1;
    // Launch kernel with 1 block of 8 threads
    factorialKernel<<<nBlocks, nThreads>>>();

       // Synchronize to ensure the kernel prints before exiting
       cudaDeviceSynchronize();

       return 0;

}
