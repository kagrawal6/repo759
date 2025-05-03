#include "matmul.cuh"
#include <cuda_runtime.h>
#include <cstdio>      // printf
#include <cstdlib>     // std::stoi
#include <iostream>    // std::cout, std::cerr
#include <string>      // std::string, std::stoi

int main(int argc, char *argv[]) {
    // Validate command-line arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> [block_dim]\n";
        return 1;
    }

    // Parse matrix size and optional block dimension
    unsigned int n = std::stoi(argv[1]);
    unsigned int block_dim = (argc > 2) ? std::stoi(argv[2]) : 16;

    // Limit matrix size to 2^14
    if (n > (1u << 14)) {
        std::cerr << "Error: matrix size must be <= 2^14\n";
        return 1;
    }

    // Prepare CUDA timing events
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;

    // ----- Integer matrix multiplication test -----
    {
        // Allocate unified memory for int matrices
        int *A, *B, *C;
        cudaMallocManaged(&A, n * n * sizeof(int));
        cudaMallocManaged(&B, n * n * sizeof(int));
        cudaMallocManaged(&C, n * n * sizeof(int));

        // Fill A and B with a simple modulo pattern
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < n; ++j) {
                A[i * n + j] = (i + j) % 10;
                B[i * n + j] = (i * j) % 10;
            }
        }

        // Create and record start event
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Execute integer tiled matmul
        matmul_1(A, B, C, n, block_dim);

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

    // ----- Float matrix multiplication test -----
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

    // ----- Double matrix multiplication test -----
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

        // Create and record start event
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Execute double tiled matmul
        matmul_3(A, B, C, n, block_dim);

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

    return 0;
}
