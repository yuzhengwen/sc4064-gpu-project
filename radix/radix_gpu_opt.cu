#include "utils.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

// This kernel calculates how many of each digit (0-9) exist in each block
__global__ void histogram_kernel(int* src, int* block_hists, int n, int exp) {
    // Shared memory for 10 decimal buckets (0-9)
    __shared__ int local_hist[10];

    // 1. Initialize shared memory to zero
    if (threadIdx.x < 10) local_hist[threadIdx.x] = 0;
    __syncthreads();

    // 2. Each thread processes elements and updates the local histogram
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        int val = src[id];
        int digit = (val < 0) ? 0 : (val / exp) % 10;
        atomicAdd(&local_hist[digit], 1); // Atomic in shared memory is very fast
    }
    __syncthreads();

    // 3. Write the block's results to the global grid-wide histogram
    // Structure: [Digit 0 counts for all blocks, Digit 1 counts for all blocks, ...]
    if (threadIdx.x < 10) {
        block_hists[threadIdx.x * gridDim.x + blockIdx.x] = local_hist[threadIdx.x];
    }
}

// This kernel uses the scanned offsets to place elements in their correct global buckets
__global__ void reorder_kernel_coalesced(int* src, int* dst, int* scanned_hists, int n, int exp) {
    // Consolidated shared memory at the top
    __shared__ int block_offsets[10];
    __shared__ int tile[256];
    __shared__ int thread_digits[256];
    __shared__ int local_offsets[256];
    __shared__ int tile_hist[10];
    __shared__ int tile_bucket_offsets[10];
    __shared__ int shuffled_tile[256];

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 1. Initialize shared memory
    if (threadIdx.x < 10) {
        block_offsets[threadIdx.x] = scanned_hists[threadIdx.x * gridDim.x + blockIdx.x];
        tile_hist[threadIdx.x] = 0;
    }
    __syncthreads(); // CRITICAL: Wait for zeros before atomics

    // 2. Load data
    int my_val = 0;
    int my_digit = 0;
    if (id < n) {
        my_val = src[id];
        my_digit = (my_val < 0) ? 0 : (my_val / exp) % 10;
        tile[threadIdx.x] = my_val;
        thread_digits[threadIdx.x] = my_digit;
        atomicAdd(&tile_hist[my_digit], 1);
    }
    __syncthreads(); // Wait for all threads to finish counting

    // 3. Intra-Block Scan (Calculated by Thread 0)
    if (threadIdx.x == 0) {
        tile_bucket_offsets[0] = 0;
        for (int d = 1; d < 10; d++) {
            tile_bucket_offsets[d] = tile_bucket_offsets[d-1] + tile_hist[d-1];
        }
    }
    __syncthreads(); // Wait for Thread 0 to finish prefix sum

    // 4. Local Shuffle
    if (id < n) {
        int local_rank = 0;
        for (int i = 0; i < threadIdx.x; i++) {
            if (thread_digits[i] == my_digit) local_rank++;
        }
        int shuffle_pos = tile_bucket_offsets[my_digit] + local_rank;
        shuffled_tile[shuffle_pos] = my_val;
    }
    __syncthreads(); // Wait for shuffle to complete

    // 5. Coalesced Write
    if (id < n) {
        int final_val = shuffled_tile[threadIdx.x];
        int final_digit = (final_val < 0) ? 0 : (final_val / exp) % 10;
        
        int final_local_rank = threadIdx.x - tile_bucket_offsets[final_digit];
        int global_pos = block_offsets[final_digit] + final_local_rank;
        dst[global_pos] = final_val;
    }
}

void gpu_parallel_radix_sort_opt(int* d_arr, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    int* d_temp;
    int* d_block_hists; // Size: 10 buckets * number of blocks
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_hists, 10 * blocksPerGrid * sizeof(int)));

    thrust::device_ptr<int> t_hists(d_block_hists);

    int max_val = find_max_host_wrapper(d_arr, n); // Helper to get max_val from device
    int* src = d_arr;
    int* dst = d_temp;

    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        // Step 1: Local Histogramming (Shared Memory)
        histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(src, d_block_hists, n, exp);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 2: Global Prefix Sum on the small histogram array
        // Much faster than scanning the entire dataset N
        thrust::exclusive_scan(t_hists, t_hists + (10 * blocksPerGrid), t_hists);

        // Step 3: Global Reorder
        reorder_kernel_coalesced<<<blocksPerGrid, threadsPerBlock>>>(src, dst, d_block_hists, n, exp);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Buffer Swap
        int* tmp = src; src = dst; dst = tmp;
    }

    if (src != d_arr) {
        CUDA_CHECK(cudaMemcpy(d_arr, src, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_temp);
    cudaFree(d_block_hists);
}