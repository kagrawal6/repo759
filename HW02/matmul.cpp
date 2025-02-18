
#include "matmul.h"
#include <cstring>   

void mmul1(const double* A, const double* B, double* C, const unsigned int n) {
    
    std::memset(C, 0, n*n*sizeof(double));

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            //double sum = 0.0;
            for (unsigned int k = 0; k < n; k++) {
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
           
        }
    }
}

void mmul2(const double* A, const double* B, double* C, const unsigned int n) {
    // Similar approach but reorder loops (i, k, j)
    // zeroing out C first
    std::memset(C, 0, n*n*sizeof(double));

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int k = 0; k < n; k++) {
            for (unsigned int j = 0; j < n; j++) {
                C[i*n + j] +=  A[i*n + k] * B[k*n + j];
            }
        }
    }
}

void mmul3(const double* A, const double* B, double* C, const unsigned int n) {
    // reorder loops to (j, k, i)
    std::memset(C, 0, n*n*sizeof(double));

    for (unsigned int j = 0; j < n; j++) {
        for (unsigned int k = 0; k < n; k++) {
            for (unsigned int i = 0; i < n; i++) {
                C[i*n + j] += A[i*n + k] * B[k*n + j] ;
            }
        }
    }
}

void mmul4(const std::vector<double>& A, const std::vector<double>& B, 
           double* C, const unsigned int n) {
    // same loop order as mmul1: (i, j, k) but vector double type
    std::memset(C, 0, n*n*sizeof(double));

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            //double sum = 0.0;
            for (unsigned int k = 0; k < n; k++) {
                C[i*n + j] += A[i*n + k] * B[k*n + j];
            }
        }
    }
}
