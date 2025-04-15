#include <cstdio>          // For printf
#include <cstdlib>         // For rand, srand
#include <ctime>           // For time (seeding RNG)
#include <cuda.h>

#include <stdio.h>
#include <random>
#include "matmul.cuh"



int main(int argc, char** argv) {

    
    // Parse CLI arguments
    size_t n = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);

    std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source()); 
	const float min = -1.0, max = 1.0; 
	std::uniform_real_distribution<float> distA(min, max);
	std::uniform_real_distribution<float> distB(min, max);

    // --- 1. Allocate host arrays (n*n)
    size_t total_elems = n * n;
    float* hA = new float[total_elems];
    float* hB = new float[total_elems];
    float* hC = new float[total_elems];

    // --- 2. Fill A, B with random floats in [-1,1]
    for (size_t i = 0; i < n * n; i++) {
		hA[i] = distA(generator);
		hB[i] = distB(generator);
	}

    // --- 3. Allocate device arrays
    float *dA, *dB, *dC;
    cudaMallocManaged((void**)&dA, total_elems * sizeof(float));
    cudaMallocManaged((void**)&dB, total_elems * sizeof(float));
    cudaMallocManaged((void**)&dC, total_elems * sizeof(float));

    // --- 4. Copy A and B to device
    cudaMemcpy(dA, hA, total_elems * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, total_elems * sizeof(float), cudaMemcpyHostToDevice);

    // --- 5. Use CUDA events to time the matmul call
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
   cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Call your matmul function
    matmul(dA, dB, dC, n, threads_per_block);

    // Record the stop event
    cudaEventRecord(stop);
    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time (ms)
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // --- 6. Copy the result C back to host
    cudaMemcpy(hC, dC, total_elems * sizeof(float), cudaMemcpyDeviceToHost);

    // --- 7. Print the last element of the resulting matrix
    // The "last element" is hC[n*n - 1]
    printf("%f\n", hC[total_elems - 1]);

    // --- 8. Print the time in ms
    printf("%f\n", elapsed_ms);

    // --- Cleanup
    delete[] hA;
    delete[] hB;
    delete[] hC;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
