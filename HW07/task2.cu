#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cuda_runtime.h>
#include "reduce.cuh"

int main(int argc, char *argv[]) {
   
    
    unsigned int N = atoi(argv[1]);
    unsigned int threads_per_block = atoi(argv[2]);
    
    // Create and fill an array of length N with random numbers in the range [-1,1] on the host
    float *h_input = (float*)malloc(N * sizeof(float));
    
    //random
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (unsigned int i = 0; i < N; i++) {
        h_input[i] = dis(gen);
    }
    
    // device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    
    // Calculate the number of blocks 
    unsigned int num_blocks = (N + 2 * threads_per_block - 1) / (2 * threads_per_block);
    cudaMalloc(&d_output, num_blocks * sizeof(float));
    
    
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start 
    cudaEventRecord(start);
    
    // Call
    reduce(&d_input, &d_output, N, threads_per_block);
    
    // Stop 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate  time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result
    float result;
    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print 
    printf("%f\n", result);
    printf("%f\n", milliseconds);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_input);
    
    return 0;
}