// task2.cpp
#include <iostream>
#include <random>
#include <chrono>
#include <cstdlib>
#include "convolution.h"


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " n m\n";
        return 1;
    }

    std::size_t n = std::atoi(argv[1]);
    std::size_t m = std::atoi(argv[2]);
    if (n == 0 || m == 0 || (m % 2 == 0)) {
        std::cerr << "n and m must be positive, and m should be odd.\n";
        return 1;
    }

     // 2. Create an n×n image in [−10, 10]
     float* image = new float[n * n];
     {
         std::random_device rd;
         std::mt19937 gen(rd());
         std::uniform_real_distribution<float> distImg(-10.0f, 10.0f);
         for (std::size_t i = 0; i < n*n; i++) {
             image[i] = distImg(gen);
         }
     }
 
     // 3. Create an m×m mask in [−1, 1]
     float* mask = new float[m * m];
     {
         std::random_device rd;
         std::mt19937 gen(rd());
         std::uniform_real_distribution<float> distMask(-1.0f, 1.0f);
         for (std::size_t i = 0; i < m*m; i++) {
             mask[i] = distMask(gen);
         }
     }

     float* output = new float[n * n];

     auto start = std::chrono::high_resolution_clock::now();
     convolve(image, output, n, mask, m);
     auto end = std::chrono::high_resolution_clock::now();
     double elapsed_ms = 
         std::chrono::duration<double, std::milli>(end - start).count();
 
     // 5. Print the time
     std::cout << elapsed_ms << "\n";
 
     // 6. Print first and last elements
     //    first element => output[0]
     //    last element  => output[n*n - 1]
     std::cout << output[0] << "\n";
     std::cout << output[n*n - 1] << "\n";
 
     // 7. Free memory
     delete[] image;
     delete[] mask;
     delete[] output;
 
     return 0;
}
