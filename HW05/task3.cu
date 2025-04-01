// task3.cu
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <random>
#include "vscale.cuh"  

int main(int argc, char** argv)
{
    // 1) Read n from command line
    if (argc < 2) {
        printf("Usage: %s <n>\n", argv[0]);
        return 1;
    }
    int n = std::atoi(argv[1]);
    if (n <= 0) {
        printf("Error: n must be a positive integer.\n");
        return 1;
    }

    // 2) Create host arrays a and b, fill them with random values
    float* a_host = new float[n];
    float* b_host = new float[n];

    // Random number generation:
    //  - a in [-10.0, 10.0]
    //  - b in [0.0,  1.0 ]
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> distA(-10.0f, 10.0f);
    std::uniform_real_distribution<float> distB(0.0f, 1.0f);

    for (int i = 0; i < n; i++) {
        a_host[i] = distA(gen);
        b_host[i] = distB(gen);
    }

    // 3) Allocate device arrays dA and dB
    float* dA = nullptr;
    float* dB = nullptr;

    cudaMalloc((void**)&dA, n * sizeof(float));
    cudaMalloc((void**)&dB, n * sizeof(float));

    // Copy data from host arrays to device arrays
    cudaMemcpy(dA, a_host, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, b_host, n * sizeof(float), cudaMemcpyHostToDevice);

    // 4) Launch vscale kernel with 16 or 512 threads per block
    int blockSize = 512;
    int gridSize  = (n + blockSize - 1) / blockSize;

    // 5) Time the kernel execution using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Call the vscale kernel. vscale is defined in vscale.cu
    vscale<<<gridSize, blockSize>>>(dA, dB, n);
    cudaEventRecord(stop);

    // Wait for kernel to finish, then measure elapsed time
    cudaEventSynchronize(stop);

    float elapsedTimeMs = 0.0f;
    cudaEventElapsedTime(&elapsedTimeMs, start, stop);

    // 6) Copy results back to b_host
    cudaMemcpy(b_host, dB, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 7) Print time, first element, and last element
    //    Each on its own line, per the assignment instructions.
    printf("%f\n", elapsedTimeMs);        // Kernel execution time in ms
    printf("%f\n", b_host[0]);            // First element of b
    printf("%f\n", b_host[n - 1]);        // Last element of b

    // Clean up
    delete[] a_host;
    delete[] b_host;

    cudaFree(dA);
    cudaFree(dB);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
