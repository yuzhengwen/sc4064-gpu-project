#pragma once
#include <iostream>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", \
        cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(err); \
    } \
} while (0)

// Utility functions (defined in main.cu)
void print_sample(const char* label, int* arr, int n, int count = 20, bool is_device = true);
void verify_and_print(int* d_arr, int n);
int find_max_host(int* h_arr, int n);

// Sorting implementations
void cpu_sequential_radix_sort(int* h_arr, int n);
void gpu_sequential_radix_sort(int* d_arr, int n);
void gpu_parallel_radix_sort(int* d_arr, int n);
void gpu_library_radix_sort(int* d_arr, int n);