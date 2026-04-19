#include "utils.h"

// Baseline parallel radix sort: per-digit predicate + scan + scatter pipeline.

// Kernel 1: Build a predicate mask for one bucket in one digit pass.
// `predicate[id] == 1` means this value belongs to the active bucket.
__global__ void evaluate_predicate_kernel(int* src, int* predicate, int n, int exp, int digit) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        predicate[id] = (((src[id] / exp) % 10) == digit) ? 1 : 0;
    }
}

// Kernel 2: Scatter values into their final range for the active bucket.
__global__ void scatter_kernel(int* src, int* dst, int* predicate, int* scan, int n, int global_offset) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        // Only threads flagged by the predicate write output this round.
        // `scan[id]` is this element's rank among flagged elements before `id`.
        if (predicate[id] == 1) {
            // `global_offset` is where this bucket starts in `dst`.
            dst[global_offset + scan[id]] = src[id];
        }
    }
}

void gpu_parallel_radix_sort(int* d_arr, int n) {
    int* d_temp;
    int* d_predicate;
    int* d_scan;
    
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_predicate, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan, n * sizeof(int)));

    int max_val = find_max_host_wrapper(d_arr, n);

    // Ping-pong buffers: each digit pass reads from `src` and writes to `dst`.
    int* src = d_arr;
    int* dst = d_temp;
    int swaps = 0;

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Outer loop: run one digit pass at a time (ones, tens, hundreds, ...).
    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        // Running base index of the current bucket inside `dst`.
        int global_offset = 0;
        
        // Inner loop: process bucket 0..9 for the current digit pass.
        for (int digit = 0; digit < 10; digit++) {
            
            // Step 1: Build a 0/1 mask for the current digit bucket.
            evaluate_predicate_kernel<<<blocksPerGrid, threadsPerBlock>>>(src, d_predicate, n, exp, digit);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Step 2: Copy into a scratch array because the scan runs in place.
            CUDA_CHECK(cudaMemcpy(d_scan, d_predicate, n * sizeof(int), cudaMemcpyDeviceToDevice));

            // Step 3: Exclusive scan converts mask -> in-bucket write indices.
            manual_exclusive_scan(d_scan, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Step 4: Scatter flagged values into this bucket's destination range.
            scatter_kernel<<<blocksPerGrid, threadsPerBlock>>>(src, dst, d_predicate, d_scan, n, global_offset);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Bucket size = last exclusive index + last predicate bit.
            // This avoids a full reduction and keeps offset tracking simple.
            int last_scan, last_pred;
            CUDA_CHECK(cudaMemcpy(&last_scan, d_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_pred, d_predicate + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
            global_offset += (last_scan + last_pred);
        }

        // Next digit pass reads from the freshly written buffer.
        int* tmp = src; src = dst; dst = tmp;
        swaps++;
    }

    // If the last pass ended in the temporary buffer, copy back once.
    if (swaps % 2 != 0) {
        CUDA_CHECK(cudaMemcpy(d_arr, src, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_temp);
    cudaFree(d_predicate);
    cudaFree(d_scan);
}