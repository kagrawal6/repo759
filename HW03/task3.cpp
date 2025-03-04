#include <iostream>
#include <cstdlib>      // for atoi, rand, RAND_MAX
#include <random>       // for std::mt19937, std::uniform_int_distribution
#include <omp.h>
#include "msort.h"      // your parallel merge sort header

int main(int argc, char* argv[]) {
    // Check the number of command-line arguments
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <n> <t> <ts>\n"
                  << "  n  = size of array\n"
                  << "  t  = number of threads\n"
                  << "  ts = threshold for recursion\n";
        return 1;
    }

    // Parse arguments
    std::size_t n  = static_cast<std::size_t>(std::atoi(argv[1]));
    int t          = std::atoi(argv[2]);
    std::size_t ts = static_cast<std::size_t>(std::atoi(argv[3]));

    // Basic validation
    if (n == 0) {
        std::cerr << "Error: n must be > 0\n";
        return 1;
    }
    if (t < 1) {
        std::cerr << "Error: t must be >= 1\n";
        return 1;
    }
    if (ts == 0) {
        std::cerr << "Error: threshold must be > 0\n";
        return 1;
    }

    // Allocate an array of length n
    int* arr = new int[n];

    // Initialize random generator to fill arr with values in [-1000, 1000]
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(-1000, 1000);

        for (std::size_t i = 0; i < n; i++) {
            arr[i] = dist(gen);
        }
    }

    // Set the number of OpenMP threads
    omp_set_num_threads(t);

    // Measure time for msort
    double start = omp_get_wtime();
    msort(arr, n, ts);  // Parallel merge sort
    double end = omp_get_wtime();
    double elapsed_ms = (end - start) * 1000.0;

    // Print results:
    //   1) First element
    //   2) Last element
    //   3) Time in milliseconds
    std::cout << arr[0]           << "\n";
    std::cout << arr[n - 1]       << "\n";
    std::cout << elapsed_ms       << "\n";

   //  for (std::size_t i = 0; i < n; i++) {
     //    std::cout << arr[i]<< "\n";
    //}

    // Clean up memory
    delete[] arr;
    return 0;
}
