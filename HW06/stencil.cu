#include <cuda_runtime.h>  // Make sure to include this for CUDA calls
#include <cstdio>          // For printf (if needed)
#include "stencil.cuh"

// 1D stencil kernel:
//   output[i] = sum_{j=-R..R} image[i + j] * mask[j+R],
// with boundary condition: image[x] = 1 if x<0 or x>=n.
//
// We store:
//   1) The mask (2*R + 1) floats in shared memory
//   2) The image block + halo (threads_per_block + 2*R) in shared memory
//
__global__
void stencil_kernel(const float* image, const float* mask, float* output,
                    unsigned int n, unsigned int r_)
{
    extern __shared__ float shMem[]; // dynamic shared memory

    int R = static_cast<int>(r_);
    int t_id = threadIdx.x;
    int t_len = blockDim.x;
    int i = blockIdx.x * t_len + t_id;  // global index for this thread

    // If out of range, do nothing
    if (i >= static_cast<int>(n)) {
        return;
    }

    // 1) First portion of shMem: the mask
    float* shared_mask = shMem;              // length (2*R + 1)

    // 2) Second portion: the block’s slice of 'image' plus halo
    float* shared_image = &shared_mask[2*R + 1]; // length (t_len + 2*R)

    // ------------------------------------------------------------
    // (a) Load the mask into shared memory cooperatively
    //     Each thread can load multiple entries, stepping by t_len
    // ------------------------------------------------------------
    for (int m = t_id; m < (2*R + 1); m += t_len) {
        shared_mask[m] = mask[m];
    }

    // ------------------------------------------------------------
    // (b) Load (t_len + 2*R) elements from the global image
    //     The leftmost element in shared_image is at (blockStart - R)
    //     If out-of-range, store 1.0f (boundary condition)
    // ------------------------------------------------------------
    int blockStart = blockIdx.x * t_len;  // start of this block in global indexing
    for (int s = t_id; s < (t_len + 2*R); s += t_len) {
        int gPos = blockStart - R + s; // global position for the s-th element in shared_image
        if (gPos < 0 || gPos >= static_cast<int>(n)) {
            shared_image[s] = 1.0f;
        } else {
            shared_image[s] = image[gPos];
        }
    }

    // Wait until all threads in the block finish loading to shared memory
    __syncthreads();

    // ------------------------------------------------------------
    // (c) Compute the stencil for global index i
    //     The center in shared_image is (t_id + R).
    //     We sum from j=-R..R => j+R in [0..2R].
    // ------------------------------------------------------------
    int sCenter = t_id + R;
    float sum = 0.0f;
    for (int j = -R; j <= R; j++) {
        int sPos = sCenter + j;     // index in shared_image
        int mPos = R + j;           // index in shared_mask
        sum += shared_image[sPos] * shared_mask[mPos];
    }

    // Write the result to output
    output[i] = sum;
}

// -------------------------------------------------------------------------
// Host function to configure & launch the kernel
// -------------------------------------------------------------------------
__host__
void stencil(const float* image, const float* mask, float* output,
             unsigned int n, unsigned int R, unsigned int threads_per_block)
{
    // Number of blocks to cover n elements
    unsigned int numBlocks = (n + threads_per_block - 1) / threads_per_block;

    // Shared memory size = (2R+1) for mask + (threads_per_block + 2R) for the data
    unsigned int shared_count = (2*R + 1) + (threads_per_block + 2*R);
    size_t shared_bytes = shared_count * sizeof(float);

    // Launch kernel
    stencil_kernel<<<numBlocks, threads_per_block, shared_bytes>>>(
        image, mask, output, n, R
    );

    // Optionally synchronize & check errors
    cudaDeviceSynchronize();
}
