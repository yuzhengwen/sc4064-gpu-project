#ifndef UTILS_H
#define UTILS_H

// Shared declarations/utilities for radix sort implementations and benchmarks.
#pragma once
#include <iostream>
#include <cuda_runtime.h>

// Centralized CUDA error checking so every allocation and copy fails loudly.
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", \
        cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(err); \
    } \
} while (0)

// Small host-side helpers used by multiple implementations.
static inline int find_max_host(int* h_arr, int n) {
    if (n <= 0) return 0;
    int max_val = h_arr[0];
    for (int i = 1; i < n; i++) {
        if (h_arr[i] > max_val) max_val = h_arr[i];
    }
    return max_val;
}

static inline int find_max_host_wrapper(int* d_arr, int n) {
    int* h_temp = new int[n];
    cudaMemcpy(h_temp, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    int res = find_max_host(h_temp, n);
    delete[] h_temp;
    return res;
}

// Shared helpers for printing, verification, scans, and the four sort variants.
void print_sample(const char* label, int* arr, int n, int count = 20, bool is_device = true);
void verify_and_print(int* d_arr, int n);
void manual_exclusive_scan(int* d_arr, int n);

// Sorting implementations used in the benchmark harness.
void cpu_sequential_radix_sort(int* h_arr, int n);
void gpu_sequential_radix_sort(int* d_arr, int n);
void gpu_parallel_radix_sort(int* d_arr, int n);
void gpu_parallel_radix_sort_opt_p2(int* d_arr, int n);
void gpu_parallel_radix_sort_opt_p3(int* d_arr, int n);
void gpu_library_radix_sort(int* d_arr, int n);

#endif