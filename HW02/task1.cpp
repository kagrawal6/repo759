#include <iostream>
#include <random>
#include <chrono>
#include <cstdlib>   // for std::atoi
#include "scan.h"


int main(int argc, char* argv[]) {
    if(argc<2){
        std::cerr<<"Usage: "<<argv[0] << "n \n";
        return 1;
    }

    std::size_t n = atoi(argv[1]);
    if (n == 0) {
        std::cerr << "n must be a positive integer.\n";
        return 1;
    }

  

       // 2. Allocate array of n floats in [-1.0, 1.0]
       float* arr = new float[n];
       std::random_device rd;
       std::mt19937 gen(rd());
       std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
       for (std::size_t i = 0; i < n; i++) {
           arr[i] = dist(gen);
       }

       float* output = new float[n];

       //running scan and measuring time
       auto start = std::chrono::high_resolution_clock::now();
       scan(arr, output, n);
       auto end = std::chrono::high_resolution_clock::now();
       double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

           // 5. Print results (time, first element, last element)
        std::cout << elapsed_ms << "\n";             //  0.06
        std::cout << output[0] << "\n";              // 0.65
        std::cout << output[n - 1] << "\n";          //  87.3

        // 6. Deallocate
        delete[] arr;
        delete[] output;

        return 0;


}
