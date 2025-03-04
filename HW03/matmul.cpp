#include "matmul.h"
#include <cstring>
#include <omp.h>

void mmul(const float* A, const float* B, float* C, const std::size_t n) {
    std::memset(C, 0, n * n * sizeof(float));

    #pragma omp parallel for // Parallelize the outer loop and avoid parallelizing inner loops
    for (std::size_t i = 0; i < n; i++) {
	#pragma omp parallel for
        for (std::size_t k = 0; k < n; k++) {
            for (std::size_t j = 0; j < n; j++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
