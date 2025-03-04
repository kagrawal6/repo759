// convolution.cpp
#include <omp.h>
#include <cstddef>
#include "convolution.h"

static inline float boundaryValue(const float *image, std::size_t n, 
                                  int x, int y) {
    // Return 0 if both x,y out of [0, n-1]
    // Return 1 if exactly one is out
    // Otherwise, return the actual pixel image[x*n + y]
    bool outX = (x < 0 || x >= (int)n);
    bool outY = (y < 0 || y >= (int)n);

    if (outX && outY) {
        return 0.0f;
    } 
    else if (outX ^ outY) {
        // ^ is logical XOR
        return 1.0f;
    } 
    else {
        // inside the domain
        return image[x * n + y];
    }
}

void convolve(const float *image, float *output, std::size_t n,
              const float *mask, std::size_t m) 
{
    int half = (m - 1) / 2;
    #pragma omp parallel for
    for (std::size_t x = 0; x < n; x++) {
        #pragma omp parallel for
        for (std::size_t y = 0; y < n; y++) {
            float accum = 0.0f;
            // Loop over the mask
            for (std::size_t i = 0; i < m; i++) {
                for (std::size_t j = 0; j < m; j++) {
                    int xx = (int)x + (int)i - half;
                    int yy = (int)y + (int)j - half;

                    float fVal = boundaryValue(image, n, xx, yy);
                    float wVal = mask[i * m + j];
                    accum += fVal * wVal;
                }
            }
            output[x * n + y] = accum;
        }
    }
}
