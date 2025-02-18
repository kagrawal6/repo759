#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>  // For floating-point comparison
#include <cstdlib>
#include "matmul.h"

const double EPSILON = 1e-6; // Tolerance for floating-point comparison

bool check_results(const double* C1, const double* C2, unsigned int n) {
    for (unsigned int i = 0; i < n * n; i++) {
        if (std::abs(C1[i] - C2[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

bool check_results(const double* C1, const std::vector<double>& C2, unsigned int n) {
    for (unsigned int i = 0; i < n * n; i++) {
        if (std::abs(C1[i] - C2[i]) > EPSILON) {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    // Matrix dimension
    unsigned int n = 1024;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Allocate and initialize matrices (both double* and std::vector versions)
    double* A = (double*)malloc(n * n * sizeof(double));
    double* B = (double*)malloc(n * n * sizeof(double));

    std::vector<double> Av;
    std::vector<double> Bv;
    Av.reserve(n * n);
    Bv.reserve(n * n);

    for (size_t i = 0; i < n * n; i++) {
        double valueA = dist(generator);
        double valueB = dist(generator);

        A[i] = valueA;
        B[i] = valueB;

        Av.push_back(valueA);
        Bv.push_back(valueB);
    }

    // Allocate memory for result matrices
    double* C = new double[n * n];   // mmul1 output
    double* C2 = new double[n * n];  // mmul2 output
    double* C3 = new double[n * n];  // mmul3 output
    double* C4 = new double[n * n]; // mmul4 output 

    // Run mmul1
    auto start = std::chrono::high_resolution_clock::now();
    mmul1(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();
    double time_mmul1 = std::chrono::duration<double, std::milli>(end - start).count();

    // Run mmul2
    start = std::chrono::high_resolution_clock::now();
    mmul2(A, B, C2, n);
    end = std::chrono::high_resolution_clock::now();
    double time_mmul2 = std::chrono::duration<double, std::milli>(end - start).count();

    // Run mmul3
    start = std::chrono::high_resolution_clock::now();
    mmul3(A, B, C3, n);
    end = std::chrono::high_resolution_clock::now();
    double time_mmul3 = std::chrono::duration<double, std::milli>(end - start).count();

    // Run mmul4 (vector-based implementation)
    start = std::chrono::high_resolution_clock::now();
    mmul4(Av, Bv, C4, n);
    end = std::chrono::high_resolution_clock::now();
    double time_mmul4 = std::chrono::duration<double, std::milli>(end - start).count();

    // Validate results
    bool mmul2_matches = check_results(C, C2, n);
    bool mmul3_matches = check_results(C, C3, n);
    bool mmul4_matches = check_results(C, C4, n);

    // Print results
    std::cout << n << "\n";
    std::cout << time_mmul1 << "\n" << C[n * n - 1] << "\n";
    std::cout << time_mmul2 << "\n" << C2[n * n - 1] << "\n";
    std::cout << time_mmul3 << "\n" << C3[n * n - 1] << "\n";
    std::cout << time_mmul4 << "\n" << C4[n * n - 1] << "\n";

    std::cout << "Validation Results:\n";
    std::cout << "mmul2 matches mmul1: " << (mmul2_matches ? "YES" : "NO") << "\n";
    std::cout << "mmul3 matches mmul1: " << (mmul3_matches ? "YES" : "NO") << "\n";
    std::cout << "mmul4 matches mmul1: " << (mmul4_matches ? "YES" : "NO") << "\n";

    // Free allocated memory
    free(A);
    free(B);
    delete[] C;
    delete[] C2;
    delete[] C3;
    delete[] C4;

    return 0;
}
