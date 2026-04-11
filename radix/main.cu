#include <iostream>
#include <vector>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <fstream> // For file output
#include "utils.h"

// Structure to hold data for the final summary table
struct Result {
    long long size;
    float cpu_ms;
    float gpu_seq_ms;
    float gpu_naive_ms;
    float gpu_lib_ms;
};

// --- CUDA KERNELS FOR DATA GENERATION & VERIFICATION ---
__global__ void setup_kernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, int *result, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        result[id] = curand(&state[id]) % 10000;
    }
}

void generate_random_data(int* d_arr, int n) {
    curandState* d_state;
    CUDA_CHECK(cudaMalloc(&d_state, n * sizeof(curandState)));
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    setup_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_state, time(NULL));
    CUDA_CHECK(cudaDeviceSynchronize());
    generate_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_state, d_arr, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_state);
}

__global__ void verify_sorted_kernel(int* data, int n, int* is_sorted) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n - 1) {
        if (data[id] > data[id + 1])
            atomicAdd(is_sorted, 1);
    }
}

int verify_sorting(int* d_arr, int n, const char* algo_name, int size_pow) {
    int* d_is_sorted;
    int h_is_sorted = 0;
    CUDA_CHECK(cudaMalloc(&d_is_sorted, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_is_sorted, &h_is_sorted, sizeof(int), cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    verify_sorted_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, n, d_is_sorted);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(&h_is_sorted, d_is_sorted, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_is_sorted);

    // SILENT ON SUCCESS: Only print if there is a failure
    if (h_is_sorted > 0) {
        printf("  ✗ CRITICAL FAILURE: %s at 2^%d failed with %d errors!\n", algo_name, size_pow, h_is_sorted);
    }
    
    return h_is_sorted;
}

// --- MAIN EXECUTION LOOP ---

int main() {
    int total_errors = 0; // Global error tracker

    // 1. Open the CORRECT file for Phase 3
    std::ofstream csvFile("phase3_results.csv");
    if (csvFile.is_open()) {
        csvFile << "Size,CPU_ms,GPU_Seq_ms,GPU_Naive_ms,GPU_Lib_ms,GPU_Opt_ms\n";
    }

    printf("--- Radix Sort Scaling Study (2^10 to 2^25) ---\n");
    printf("Running 80 performance tests... (Silent verification unless error occurs)\n\n");
    printf("Log: Standard Output | Data: phase3_results.csv\n\n");

    for (int i = 10; i <= 25; i++) {
        long long SIZE = 1LL << i;
        
        printf("[%02d] Dataset Size: 2^%d (%lld elements)\n", i, i, SIZE);

        // Memory Allocation & Data Generation
        int* h_original = new int[SIZE];
        int* h_test = new int[SIZE];
        int* d_original;
        int* d_test;
        CUDA_CHECK(cudaMalloc(&d_original, SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_test, SIZE * sizeof(int)));

        generate_random_data(d_original, (int)SIZE);
        CUDA_CHECK(cudaMemcpy(h_original, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToHost));

        // --- UPDATED WARMUP (Now with Initialization) ---
        int* d_warmup;
        CUDA_CHECK(cudaMalloc(&d_warmup, 100 * sizeof(int)));
        
        // CRITICAL: Initialize warmup data so kernels don't process negative garbage
        CUDA_CHECK(cudaMemset(d_warmup, 0, 100 * sizeof(int))); 

        gpu_sequential_radix_sort(d_warmup, 100);
        gpu_parallel_radix_sort(d_warmup, 100);
        gpu_parallel_radix_sort_opt(d_warmup, 100);
        gpu_library_radix_sort(d_warmup, 100);
        cudaFree(d_warmup);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // [TEST 1: CPU]
        memcpy(h_test, h_original, SIZE * sizeof(int));
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_sequential_radix_sort(h_test, (int)SIZE);
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        float cpu_ms = std::chrono::duration<float, std::milli>(cpu_stop - cpu_start).count();
        CUDA_CHECK(cudaMemcpy(d_test, h_test, SIZE * sizeof(int), cudaMemcpyHostToDevice));
        total_errors += verify_sorting(d_test, (int)SIZE, "CPU Seq", i);
        printf("     - CPU Seq:     %8.3f ms | %8.2f M-elem/s\n", cpu_ms, (SIZE / (cpu_ms * 1000.0f)));

        // [TEST 2: GPU Seq]
        CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaEventRecord(start);
        gpu_sequential_radix_sort(d_test, (int)SIZE);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float seq_ms; cudaEventElapsedTime(&seq_ms, start, stop);
        total_errors += verify_sorting(d_test, (int)SIZE, "GPU Seq", i);
        printf("     - GPU Seq:     %8.3f ms | %8.2f M-elem/s\n", seq_ms, (SIZE / (seq_ms * 1000.0f)));

        // [TEST 3: GPU Naive]
        CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaEventRecord(start);
        gpu_parallel_radix_sort(d_test, (int)SIZE);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float par_ms; cudaEventElapsedTime(&par_ms, start, stop);
        total_errors += verify_sorting(d_test, (int)SIZE, "GPU Naive", i);
        printf("     - GPU Naive:   %8.3f ms | %8.2f M-elem/s\n", par_ms, (SIZE / (par_ms * 1000.0f)));

        // [TEST 4: GPU Thrust]
        CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaEventRecord(start);
        gpu_library_radix_sort(d_test, (int)SIZE);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float lib_ms; cudaEventElapsedTime(&lib_ms, start, stop);
        total_errors += verify_sorting(d_test, (int)SIZE, "GPU Thrust", i);
        printf("     - GPU Thrust:  %8.3f ms | %8.2f M-elem/s\n", lib_ms, (SIZE / (lib_ms * 1000.0f)));

        // [TEST 5: GPU Optimized]
        CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaEventRecord(start);
        gpu_parallel_radix_sort_opt(d_test, (int)SIZE);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float opt_ms; cudaEventElapsedTime(&opt_ms, start, stop);
        total_errors += verify_sorting(d_test, (int)SIZE, "GPU Opt", i);
        printf("     - GPU Opt:     %8.3f ms | %8.2f M-elem/s\n", opt_ms, (SIZE / (opt_ms * 1000.0f)));

        // Save data to CSV
        if (csvFile.is_open()) {
            csvFile << SIZE << "," << cpu_ms << "," << seq_ms << "," << par_ms << "," << lib_ms << "," << opt_ms << "\n";
            csvFile.flush(); 
        }

        // Cleanup current size
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(d_original); cudaFree(d_test);
        delete[] h_original; delete[] h_test;
        printf("\n");
    }

    // --- FINAL SUMMARY ---
    printf("\n==========================================\n");
    if (total_errors == 0) {
        printf("  FINAL STATUS: ALL CLEAR ✓\n");
        printf("  All 80 sorting tests passed successfully.\n");
    } else {
        printf("  FINAL STATUS: %d TOTAL ERRORS FOUND ✗\n", total_errors);
        printf("  Check the log above for specific failure points.\n");
    }
    printf("==========================================\n");

    csvFile.close();
    printf("Evaluation complete. Results saved to phase3_results.csv\n");

    return 0;
}