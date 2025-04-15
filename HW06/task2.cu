#include <cstdio>      // For printf
#include <cstdlib>     // For rand, srand
#include <ctime>       // For time(nullptr)
#include <stdio.h>
#include <random>
#include <cuda.h>

#include "stencil.cuh" 




int main(int argc, char** argv) {
  

    // Parse command-line arguments
    unsigned int n                 = (atoi(argv[1]));
    unsigned int R                 = (atoi(argv[2]));
    unsigned int threads_per_block = (atoi(argv[3]));

    std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source()); 
	const int min = -1.0, max = 1.0; 
	std::uniform_real_distribution<float> dist(min, max);

    // 1. Allocate host arrays
    float* hImage  = new float[n];
    float* hMask   = new float[2*R + 1];
    float* hOutput = new float[n];

	for (size_t i = 0; i < n; i++) {
        hImage[i] = dist(generator);
    }
  
  for (size_t i = 0; i < (2*R + 1); i++) {
      hMask[i] = dist(generator);
    }

 

    // 3. Allocate device arrays
    float *dImage, *dMask, *dOutput;

    cudaMallocManaged((void**)&dImage, n * sizeof(float));
    cudaMallocManaged((void**)&dMask, (2*R + 1) * sizeof(float));
    cudaMallocManaged((void**)&dOutput, n * sizeof(float));

    // 4. Copy data from host to device
    cudaMemcpy(dImage, hImage, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dMask, hMask, (2*R + 1)*sizeof(float), cudaMemcpyHostToDevice);

    // 5. Time the stencil call using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start);

    // Call your stencil function (defined in stencil.cu)
    stencil(dImage, dMask, dOutput, n, R, threads_per_block);

    // Record stop + sync
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 6. Copy the result back to host
    cudaMemcpy(hOutput, dOutput, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 7. Print the last element of the resulting output array
    //    The assignment says "Print the last element" => hOutput[n-1]
    printf("%f\n", hOutput[n - 1]);

    // 8. Print the time in ms
    printf("%f\n", elapsed_ms);

    // Cleanup
    delete[] hImage;
    delete[] hMask;
    delete[] hOutput;

    cudaFree(dImage);
    cudaFree(dMask);
    cudaFree(dOutput);

    return 0;
}
