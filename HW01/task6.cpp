#include <iostream>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {

    // Convert command line argument to integer
    int N = std::atoi(argv[1]);

    // Print numbers from 0 to N using printf
    for (int i = 0; i <= N; ++i) {
        std::printf("%d ", i);
    }
    std::printf("\n");

    // Print numbers from N to 0 using std::cout
    for (int i = N; i >= 0; --i) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
