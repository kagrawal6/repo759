#include "matmul.cuh"
#include <cuda_runtime.h>
#include <cstdio>      
#include <cstdlib>     
#include <iostream>    
#include <string>     

int main(int argc, char *argv[]) {
   

    
    unsigned int n = std::stoi(argv[1]);
    unsigned int block_dim = (argc > 2) ? std::stoi(argv[2]) : 16;

    // Limit matrix size to 2^14
    if (n > (1u << 14)) {
        std::cerr << "matrix size must be <= 2^14\n";
        return 1;
    }

   
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;

    //  Integer 
    {
        //  int matrices
        int *A, *B, *C;
        cudaMallocManaged(&A, n * n * sizeof(int));
        cudaMallocManaged(&B, n * n * sizeof(int));
        cudaMallocManaged(&C, n * n * sizeof(int));

        
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < n; ++j) {
                A[i * n + j] = (i + j) % 10;
                B[i * n + j] = (i * j) % 10;
            }
        }

        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Execute integer 
        matmul_1(A, B, C, n, block_dim);

        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Measure elapsed time
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Output first element, last element, and runtime
        std::cout << C[0] << "\n"
                  << C[n * n - 1] << "\n"
                  << milliseconds << "\n";

        // Clean up events and memory
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }

    // Float
    {
        // Allocate unified memory for float matrices
        float *A, *B, *C;
        cudaMallocManaged(&A, n * n * sizeof(float));
        cudaMallocManaged(&B, n * n * sizeof(float));
        cudaMallocManaged(&C, n * n * sizeof(float));

        // Fill A and B with fractional patterns
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < n; ++j) {
                A[i * n + j] = ((i + j) % 10) / 10.0f;
                B[i * n + j] = ((i * j) % 10) / 10.0f;
            }
        }

        // Create and record start event
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Execute float tiled matmul
        matmul_2(A, B, C, n, block_dim);

        // Record stop event and wait for kernel finish
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Measure elapsed time
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Output first element, last element, and runtime
        std::cout << C[0] << "\n"
                  << C[n * n - 1] << "\n"
                  << milliseconds << "\n";

        // Clean up events and memory
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }

    // Double
    {
        // Allocate unified memory for double matrices
        double *A, *B, *C;
        cudaMallocManaged(&A, n * n * sizeof(double));
        cudaMallocManaged(&B, n * n * sizeof(double));
        cudaMallocManaged(&C, n * n * sizeof(double));

        // Fill A and B with fractional patterns
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < n; ++j) {
                A[i * n + j] = ((i + j) % 10) / 10.0;
                B[i * n + j] = ((i * j) % 10) / 10.0;
            }
        }

        
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        
        matmul_3(A, B, C, n, block_dim);

       
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Measure elapsed time
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Output first element, last element, and runtime
        std::cout << C[0] << "\n"
                  << C[n * n - 1] << "\n"
                  << milliseconds << "\n";

        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
    }

    return 0;
}
