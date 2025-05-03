#include "matmul.cuh"
#include <cuda_runtime.h>


template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C, unsigned int n) {
    
    const unsigned int TILE_DIM = blockDim.x;
    
    
    extern __shared__ char shared_mem[];
    T* tile_A = (T*)shared_mem;
    T* tile_B = (T*)&shared_mem[TILE_DIM * TILE_DIM * sizeof(T)];
    
    
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    
   
    unsigned int row = blockIdx.y * TILE_DIM + ty;
    unsigned int col = blockIdx.x * TILE_DIM + tx;
    
    
    T sum = 0;
    
    
    for (unsigned int t = 0; t < (n + TILE_DIM - 1) / TILE_DIM; t++) {
        
        if (row < n && t * TILE_DIM + tx < n) {
            tile_A[ty * TILE_DIM + tx] = A[row * n + t * TILE_DIM + tx];
        } else {
            tile_A[ty * TILE_DIM + tx] = 0;
        }
        
        if (t * TILE_DIM + ty < n && col < n) {
            tile_B[ty * TILE_DIM + tx] = B[(t * TILE_DIM + ty) * n + col];
        } else {
            tile_B[ty * TILE_DIM + tx] = 0;
        }
        
        
        __syncthreads();
        
        // Compute partial 
        for (unsigned int k = 0; k < TILE_DIM; k++) {
            sum += tile_A[ty * TILE_DIM + k] * tile_B[k * TILE_DIM + tx];
        }
        
        // Synchronize 
        __syncthreads();
    }
    
    // Write  global memory
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Host function
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                     unsigned int block_dim) {
    //  grid dimensions
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    
    // shared memory size (two tiles)
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(int);
    
    //  kernel
    matmul_kernel<int><<<grid, block, shared_mem_size>>>(A, B, C, n);
    
   
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                      unsigned int block_dim) {
    //  grid dimensions
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    
    // Calculate shared memory size (two tiles)
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(float);
    
    // Launch kernel
    matmul_kernel<float><<<grid, block, shared_mem_size>>>(A, B, C, n);
    
    // Synchronize 
    cudaDeviceSynchronize();
}

__host__ void matmul_3(const double *A, const double *B, double *C,
                     unsigned int n, unsigned int block_dim) {
    // grid dimensions
    dim3 block(block_dim, block_dim);
    dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    
    
    size_t shared_mem_size = 2 * block_dim * block_dim * sizeof(double);
    
    //  kernel
    matmul_kernel<double><<<grid, block, shared_mem_size>>>(A, B, C, n);
    
    // Synchronize 
    cudaDeviceSynchronize();
}