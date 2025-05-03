#include "reduce.cuh"

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    
   
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    
    
    float my_val = 0.0f;
    
    
    if (i < n) {
        my_val = g_idata[i];
    }
    
    
    if (i + blockDim.x < n) {
        my_val += g_idata[i + blockDim.x];
    }
    
    // Store result in shared memory
    sdata[tid] = my_val;
    __syncthreads();
    
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
   
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

__host__ void reduce(float **input, float **output, unsigned int N,
                    unsigned int threads_per_block) {
    unsigned int current_size = N;
    float *current_input = *input;
    float *current_output = *output;
    
    while (current_size > 1) {
        //  number of blocks needed
        unsigned int num_blocks = (current_size + 2 * threads_per_block - 1) / (2 * threads_per_block);
        
        //  kernel
        reduce_kernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
            current_input, current_output, current_size);
        
        //  next iteration
        current_size = num_blocks;
        
        // Swap 
        if (current_size > 1) {
            float *temp = current_input;
            current_input = current_output;
            current_output = temp;
        }
    }
    
    // Copy 
    if (current_output != *input) {
        cudaMemcpy(*input, current_output, sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    // Synchronize
    cudaDeviceSynchronize();
}