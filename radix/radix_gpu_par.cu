#include "utils.h"
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

// Kernel 1: Evaluate Predicate
// Maps elements to 1 if they belong in the current digit bucket, or 0 if they do not.
__global__ void evaluate_predicate_kernel(int* src, int* predicate, int n, int exp, int digit) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        predicate[id] = (((src[id] / exp) % 10) == digit) ? 1 : 0;
    }
}

// Kernel 2: Scatter Elements
// Places the elements into the new array using the exact index calculated by the scan.
__global__ void scatter_kernel(int* src, int* dst, int* predicate, int* scan, int n, int global_offset) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        // Only move the element if it belongs in the current digit bucket
        if (predicate[id] == 1) {
            // scan[id] contains the exact number of matching elements that appeared before this one.
            // Adding global_offset shifts it past the buckets we have already filled.
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

    // Find max value to know how many digit passes we need
    int* h_arr = new int[n];
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));
    int max_val = find_max_host(h_arr, n);
    delete[] h_arr;

    int* src = d_arr;
    int* dst = d_temp;
    int swaps = 0;

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Wrap raw pointers in thrust::device_ptr so the Thrust library can read them
    thrust::device_ptr<int> t_predicate(d_predicate);
    thrust::device_ptr<int> t_scan(d_scan);

    // Outer Loop: Move through the digits (1s, 10s, 100s, etc.)
    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        int global_offset = 0;
        
        // Inner Loop: Process each of the 10 possible digit values sequentially
        for (int digit = 0; digit < 10; digit++) {
            
            // Step 1: Flag the elements that have this specific digit
            evaluate_predicate_kernel<<<blocksPerGrid, threadsPerBlock>>>(src, d_predicate, n, exp, digit);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Step 2: Parallel Prefix Sum (Exclusive Scan)
            // This calculates the deterministic output index for every flagged element
            thrust::exclusive_scan(t_predicate, t_predicate + n, t_scan);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Step 3: Scatter the flagged elements to their final positions
            scatter_kernel<<<blocksPerGrid, threadsPerBlock>>>(src, dst, d_predicate, d_scan, n, global_offset);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Update the global offset for the next digit bucket.
            // The total elements placed in this bucket equals the scan index of the very last element 
            // PLUS the predicate value of the very last element.
            int last_scan, last_pred;
            CUDA_CHECK(cudaMemcpy(&last_scan, d_scan + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_pred, d_predicate + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
            global_offset += (last_scan + last_pred);
        }

        // Swap the source and destination pointers for the next exponent pass
        int* tmp = src; src = dst; dst = tmp;
        swaps++;
    }

    // If we did an odd number of swaps, the final sorted data is sitting in d_temp.
    // Copy it back to the original array.
    if (swaps % 2 != 0) {
        CUDA_CHECK(cudaMemcpy(d_arr, src, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_temp);
    cudaFree(d_predicate);
    cudaFree(d_scan);
}