// task2.cpp
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include "convolution.h"


int main(int argc, char* argv[]) {
    // if (argc < 3) {
    //     std::cerr << "Usage: " << argv[0] << " n m\n";
    //     return 1;
    // }

    std::size_t n = std::atoi(argv[1]);
    std::size_t t = std::atoi(argv[2]);
    // if (n == 0 || m == 0 || (m % 2 == 0)) {
    //     std::cerr << "n and m must be positive, and m should be odd.\n";
    //     return 1;
    // }

     // 2. Create an n×n image in [−10, 10]

     omp_set_num_threads(t);

     float* image = new float[n * n];
     {
         std::random_device rd;
         std::mt19937 gen(rd());
         std::uniform_real_distribution<float> distImg(-10.0f, 10.0f);
         for (std::size_t i = 0; i < n*n; i++) {
             image[i] = distImg(gen);
         }
     }
 
     // 3. Create an 3×3 mask in [−1, 1]
     float* mask = new float[3 * 3];
     {
         std::random_device rd;
         std::mt19937 gen(rd());
         std::uniform_real_distribution<float> distMask(-1.0f, 1.0f);
         for (std::size_t i = 0; i < 3*3; i++) {
             mask[i] = distMask(gen);
         }
     }

     float* output = new float[n * n];

    
    double start = omp_get_wtime();
    convolve(image, output, n, mask, 3);
    double end = omp_get_wtime();
    double time_taken = (end - start) * 1000.0; // Convert to milliseconds


 
 

     // Print first and last elements and time
   
     std::cout << output[0] << "\n";
     std::cout << output[n*n - 1] << "\n";
     std::cout << time_taken << "\n";
 
     //  Free memory
     delete[] image;
     delete[] mask;
     delete[] output;
 
     return 0;
}