#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "matmul.h"

int main(int argc, char *argv[]) {
    // 1. matrix dimension n 
    unsigned int n = 1024; 
   

    // 2. Generate A and B std::vector<double>
    std::vector<double> A_vec(n*n), B_vec(n*n);
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (unsigned int i = 0; i < n*n; i++) {
            A_vec[i] = dist(gen);
            B_vec[i] = dist(gen);
        }
    }

    // A_raw and B_raw double* version for mmul1,2,3
    double* A_raw = new double[n*n];
    double* B_raw = new double[n*n];

    // Copy data from A_vec, B_vec to A_raw, B_raw
    for (unsigned int i = 0; i < n*n; i++) {
        A_raw[i] = A_vec[i];
        B_raw[i] = B_vec[i];
    }

    //  C arrays
    double* C = new double[n*n];
    double* C2 = new double[n*n];
    double* C3 = new double[n*n];
    double* C4 = new double[n*n];

    //mmul1
    auto start = std::chrono::high_resolution_clock::now();
    mmul1(A_raw, B_raw, C, n);
    auto end = std::chrono::high_resolution_clock::now();
    double time_mmul1 = std::chrono::duration<double, std::milli>(end - start).count();

    //mmul2
    start = std::chrono::high_resolution_clock::now();
    mmul2(A_raw, B_raw, C2, n);
    end = std::chrono::high_resolution_clock::now();
    double time_mmul2 = std::chrono::duration<double, std::milli>(end - start).count();

    // mmul3
    start = std::chrono::high_resolution_clock::now();
    mmul3(A_raw, B_raw, C3, n);
    end = std::chrono::high_resolution_clock::now();
    double time_mmul3 = std::chrono::duration<double, std::milli>(end - start).count();

    //mmul4 (vector version)
    start = std::chrono::high_resolution_clock::now();
    mmul4(A_vec, B_vec, C4, n);
    end = std::chrono::high_resolution_clock::now();
    double time_mmul4 = std::chrono::duration<double, std::milli>(end - start).count();

    // Print results
    // - n
    // - mmul1 time, 
    // - last element of C
    // - mmul2 time 
    // - last element of C2
    // - mmul3 time 
    // - last element of C3
    // - mmul4 time 
    // - last element of C4

    std::cout << n << "\n";
    std::cout << time_mmul1 << "\n" << C[n*n - 1] << "\n";
    std::cout << time_mmul2 << "\n" << C2[n*n - 1] << "\n";
    std::cout << time_mmul3 << "\n" << C3[n*n - 1] << "\n";
    std::cout << time_mmul4 << "\n" << C4[n*n - 1] << "\n";

   
    delete[] A_raw;
    delete[] B_raw;
    delete[] C;
    delete[] C2;
    delete[] C3;
    delete[] C4;

    return 0;



}
