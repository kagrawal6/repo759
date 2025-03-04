#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include "matmul.h"


int main(int argc, char* argv[]) {
    // if (argc != 3) {
    //     std::cerr << "Usage: " << argv[0] << " <n> <t>\n";
    //     return 1;
    // }

    std::size_t n = atoi(argv[1]); // Convert input to integer
    int t = atoi(argv[2]);

        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        float* A = (float*)malloc(n * n * sizeof(float));
        float* B = (float*)malloc(n * n * sizeof(float));
    
        // Set number of threadss
        omp_set_num_threads(t);

        // Initialize matrices A and B
    
    for (std::size_t i = 0; i < n * n; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    float* C = new float[n * n];

    // Measure time using OpenMP timing function
    double start = omp_get_wtime();
    mmul(A,B,C,n);
    double end = omp_get_wtime();
    
    double time_taken = (end - start) * 1000.0; // Convert to milliseconds

    // Print required outputs
    std::cout << C[0] << "\n";          // First element of C
    std::cout << C[n * n - 1] << "\n";  // Last element of C
    std::cout << time_taken << "\n";    // Time taken in ms

    free(A);
    free(B);
    delete[] C;

    return 0;
}
