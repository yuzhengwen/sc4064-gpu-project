#include "utils.h"

// =========================================================================
// --- RADIX SORT PHASE 2 KERNELS ---
// =========================================================================

__global__ void histogram_kernel_p2(int* src, int* block_hists, int n, int exp) {
    __shared__ int local_hist[10];

    if (threadIdx.x < 10) local_hist[threadIdx.x] = 0;
    __syncthreads();

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        int val = src[id];
        int digit = (val < 0) ? 0 : (val / exp) % 10;
        atomicAdd(&local_hist[digit], 1); 
    }
    __syncthreads();

    if (threadIdx.x < 10) {
        block_hists[threadIdx.x * gridDim.x + blockIdx.x] = local_hist[threadIdx.x];
    }
}

__global__ void reorder_kernel_p2(int* src, int* dst, int* scanned_hists, int n, int exp) {
    __shared__ int block_offsets[10];
    __shared__ int thread_digits[256]; 

    if (threadIdx.x < 10) {
        block_offsets[threadIdx.x] = scanned_hists[threadIdx.x * gridDim.x + blockIdx.x];
    }
    
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int my_digit = -1;
    if (id < n) {
        int val = src[id];
        my_digit = (val < 0) ? 0 : (val / exp) % 10;
        thread_digits[threadIdx.x] = my_digit;
    } else {
        thread_digits[threadIdx.x] = -1;
    }
    __syncthreads(); 

    if (id < n) {
        int local_rank = 0;
        for (int i = 0; i < threadIdx.x; i++) {
            if (thread_digits[i] == my_digit) {
                local_rank++;
            }
        }
        int final_pos = block_offsets[my_digit] + local_rank;
        dst[final_pos] = src[id];
    }
}

void gpu_parallel_radix_sort_opt_p2(int* d_arr, int n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    int* d_temp;
    int* d_block_hists; 
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_block_hists, 10 * blocksPerGrid * sizeof(int)));

    int max_val = find_max_host_wrapper(d_arr, n); 
    int* src = d_arr;
    int* dst = d_temp;

    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        histogram_kernel_p2<<<blocksPerGrid, threadsPerBlock>>>(src, d_block_hists, n, exp);
        CUDA_CHECK(cudaDeviceSynchronize());

        // COMPLETELY MANUAL SCAN (No Thrust!)
        manual_exclusive_scan(d_block_hists, 10 * blocksPerGrid);
        CUDA_CHECK(cudaDeviceSynchronize());

        reorder_kernel_p2<<<blocksPerGrid, threadsPerBlock>>>(src, dst, d_block_hists, n, exp);
        CUDA_CHECK(cudaDeviceSynchronize());

        int* tmp = src; src = dst; dst = tmp;
    }

    if (src != d_arr) {
        CUDA_CHECK(cudaMemcpy(d_arr, src, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_temp);
    cudaFree(d_block_hists);
}