#include "utils.h"

// Phase-3 optimized radix variant: tile-level regrouping for more coalesced writes.

// =========================================================================
// --- RADIX SORT PHASE 3 KERNELS (TILE SHUFFLE FOR COALESCED WRITES) ---
// =========================================================================

__global__ void histogram_kernel_p3(int* src, int* block_hists, int n, int exp) {
    __shared__ int local_hist[10];

    // Same per-block histogram stage as phase 2 for each digit pass.
    if (threadIdx.x < 10) local_hist[threadIdx.x] = 0;
    __syncthreads();

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        int val = src[id];
        // Negative handling is clamped to bucket 0 in this implementation.
        int digit = (val < 0) ? 0 : (val / exp) % 10;
        atomicAdd(&local_hist[digit], 1); 
    }
    __syncthreads();

    if (threadIdx.x < 10) {
        block_hists[threadIdx.x * gridDim.x + blockIdx.x] = local_hist[threadIdx.x];
    }
}

__global__ void reorder_kernel_coalesced_p3(int* src, int* dst, int* scanned_hists, int n, int exp) {
    __shared__ int block_offsets[10];
    __shared__ int tile[256];
    __shared__ int thread_digits[256];
    // NOTE: `local_offsets` is reserved for experimentation and is currently unused.
    __shared__ int local_offsets[256];
    __shared__ int tile_hist[10];
    __shared__ int tile_bucket_offsets[10];
    __shared__ int shuffled_tile[256];

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Base output offsets for each bucket in this block.
    if (threadIdx.x < 10) {
        block_offsets[threadIdx.x] = scanned_hists[threadIdx.x * gridDim.x + blockIdx.x];
        tile_hist[threadIdx.x] = 0;
    }
    __syncthreads(); 

    int my_val = 0;
    int my_digit = 0;
    if (id < n) {
        // Stage this block's values and bucket IDs into shared memory.
        my_val = src[id];
        my_digit = (my_val < 0) ? 0 : (my_val / exp) % 10;
        tile[threadIdx.x] = my_val;
        thread_digits[threadIdx.x] = my_digit;
        atomicAdd(&tile_hist[my_digit], 1);
    }
    __syncthreads(); 

    // Compute per-bucket starting offsets inside the shared-memory tile.
    if (threadIdx.x == 0) {
        tile_bucket_offsets[0] = 0;
        for (int d = 1; d < 10; d++) {
            tile_bucket_offsets[d] = tile_bucket_offsets[d-1] + tile_hist[d-1];
        }
    }
    __syncthreads(); 

    if (id < n) {
        int local_rank = 0;
        // Kept for potential warp-level rank optimization; currently unused.
        unsigned int mask = __ballot_sync(0xFFFFFFFF, true); 
        
        for (int i = 0; i < threadIdx.x; i++) {
            if (thread_digits[i] == my_digit) local_rank++;
        }
        
        // Reorder values in shared memory so equal-bucket values become contiguous.
        int shuffle_pos = tile_bucket_offsets[my_digit] + local_rank;
        shuffled_tile[shuffle_pos] = my_val;
    }
    __syncthreads(); 

    if (id < n) {
        // Read in thread order after tile shuffle; data order now encodes bucket grouping.
        int final_val = shuffled_tile[threadIdx.x];
        int final_digit = (final_val < 0) ? 0 : (final_val / exp) % 10;
        
        // Convert tile index back to rank inside the selected bucket.
        int final_local_rank = threadIdx.x - tile_bucket_offsets[final_digit];
        // Global destination index = scanned block offset + tile-local bucket rank.
        int global_pos = block_offsets[final_digit] + final_local_rank;
        dst[global_pos] = final_val;
    }
}

void gpu_parallel_radix_sort_opt_p3(int* d_arr, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    int* d_temp;
    int* d_block_hists; 
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_hists, 10 * blocksPerGrid * sizeof(int)));

    int max_val = find_max_host_wrapper(d_arr, n); 
    // Ping-pong buffers across digit passes.
    int* src = d_arr;
    int* dst = d_temp;

    // One digit pass per loop iteration.
    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        histogram_kernel_p3<<<blocksPerGrid, threadsPerBlock>>>(src, d_block_hists, n, exp);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Exclusive scan converts per-block counts into absolute output offsets.
        manual_exclusive_scan(d_block_hists, 10 * blocksPerGrid);
        CUDA_CHECK(cudaDeviceSynchronize());

        reorder_kernel_coalesced_p3<<<blocksPerGrid, threadsPerBlock>>>(src, dst, d_block_hists, n, exp);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap ping-pong roles for the next digit pass.
        int* tmp = src; src = dst; dst = tmp;
    }

    if (src != d_arr) {
        CUDA_CHECK(cudaMemcpy(d_arr, src, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_temp);
    cudaFree(d_block_hists);
}