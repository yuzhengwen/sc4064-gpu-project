#include "utils.h"

// Kernel 1: Evaluate Predicate
// Maps elements to 1 if they belong in the current digit bucket, or 0 if they do not.
__global__ void evaluate_predicate_kernel(int* src, int* predicate, int n, int exp, int digit) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        predicate[id] = (((src[id] / exp) % 10) == digit) ? 1 : 0;
    }
}

// Kernel 2: Scatter Elements
__global__ void scatter_kernel(int* src, int* dst, int* predicate, int* scan, int n, int global_offset) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        // Only move the element if it belongs in the current digit bucket
        // We still need the ORIGINAL predicate array here!
        if (predicate[id] == 1) {
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

    int* src = d_arr;
    int* dst = d_temp;
    int swaps = 0;

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Outer Loop: Move through the digits
    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        int global_offset = 0;
        
        // Inner Loop: Process each digit bucket
        for (int digit = 0; digit < 10; digit++) {
            
            // Step 1: Flag the elements
            evaluate_predicate_kernel<<<blocksPerGrid, threadsPerBlock>>>(src, d_predicate, n, exp, digit);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Step 2: Copy predicate to scan array (CRITICAL FIX)
            // Because our manual scan is IN-PLACE, we must copy the 1s and 0s 
            // into d_scan first so we don't destroy d_predicate.
            CUDA_CHECK(cudaMemcpy(d_scan, d_predicate, n * sizeof(int), cudaMemcpyDeviceToDevice));

            // Step 3: Completely Manual Prefix Sum (Exclusive Scan)
            manual_exclusive_scan(d_scan, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Step 4: Scatter the flagged elements
            scatter_kernel<<<blocksPerGrid, threadsPerBlock>>>(src, dst, d_predicate, d_scan, n, global_offset);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Update the global offset for the next digit bucket.
            int last_scan, last_pred;
            CUDA_CHECK(cudaMemcpy(&last_scan, d_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_pred, d_predicate + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
            global_offset += (last_scan + last_pred);
        }

        int* tmp = src; src = dst; dst = tmp;
        swaps++;
    }

    if (swaps % 2 != 0) {
        CUDA_CHECK(cudaMemcpy(d_arr, src, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_temp);
    cudaFree(d_predicate);
    cudaFree(d_scan);
}