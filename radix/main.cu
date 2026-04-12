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

    if (h_is_sorted > 0) {
        printf("   ✗ CRITICAL FAILURE: %s at 2^%d failed with %d errors!\n", algo_name, size_pow, h_is_sorted);
    }
    
    return h_is_sorted;
}

// Thorough Burn-in for all GPU implementations
void thorough_gpu_warmup(int n_warmup = 100000, int iterations = 10) {
    printf("Performing thorough GPU burn-in (%d elements, %d iterations)...\n", n_warmup, iterations);
    int* d_warmup;
    CUDA_CHECK(cudaMalloc(&d_warmup, n_warmup * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_warmup, 0, n_warmup * sizeof(int)));

    for (int i = 0; i < iterations; i++) {
        gpu_sequential_radix_sort(d_warmup, n_warmup);
        gpu_parallel_radix_sort(d_warmup, n_warmup);
        gpu_parallel_radix_sort_opt_p2(d_warmup, n_warmup);
        gpu_parallel_radix_sort_opt_p3(d_warmup, n_warmup);
        gpu_library_radix_sort(d_warmup, n_warmup);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaFree(d_warmup));
    printf("Burn-in complete. Hardware is now at peak performance state.\n\n");
}

__global__ void force_context_init() {
    __shared__ int dummy[1024];
    if (threadIdx.x < 1024) dummy[threadIdx.x] = threadIdx.x;
}

// --- MAIN EXECUTION LOOP ---

int main() {
    int total_errors = 0;
    const int NUM_TRIALS = 3;
    
    force_context_init<<<1, 1024>>>();
    cudaDeviceSynchronize();

    // 1. Hardware Burn-in
    thorough_gpu_warmup();

    // 2. Open CSV with updated headers for comparison
    std::ofstream csvFile("phase4_results.csv");
    if (csvFile.is_open()) {
        csvFile << "Size,CPU_ms,GPU_Seq_ms,GPU_Naive_ms,GPU_P2_Opt_ms,GPU_P3_Opt_ms,GPU_Thrust_ms\n";
    }

    printf("--- Radix Sort Scaling Study (Phase 4 - Unified Comparison) ---\n");
    printf("Log: Standard Output | Data: phase4_results.csv\n\n");

    for (int i = 10; i <= 25; i++) {
        long long SIZE = 1LL << i;
        printf("[%02d] Dataset Size: 2^%d (%lld elements)\n", i, i, SIZE);

        float t_cpu = 0, t_seq = 0, t_naive = 0, t_p2 = 0, t_p3 = 0, t_lib = 0;

        for (int t = 0; t < NUM_TRIALS; t++) {
            int* h_original = new int[SIZE];
            int* h_test = new int[SIZE];
            int* d_original;
            int* d_test;
            CUDA_CHECK(cudaMalloc(&d_original, SIZE * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_test, SIZE * sizeof(int)));

            generate_random_data(d_original, (int)SIZE);
            CUDA_CHECK(cudaMemcpy(h_original, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToHost));

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // [TEST 1: CPU]
            memcpy(h_test, h_original, SIZE * sizeof(int));
            auto cpu_start = std::chrono::high_resolution_clock::now();
            cpu_sequential_radix_sort(h_test, (int)SIZE);
            auto cpu_stop = std::chrono::high_resolution_clock::now();
            t_cpu += std::chrono::duration<float, std::milli>(cpu_stop - cpu_start).count();
            if (t == 0) {
                CUDA_CHECK(cudaMemcpy(d_test, h_test, SIZE * sizeof(int), cudaMemcpyHostToDevice));
                total_errors += verify_sorting(d_test, (int)SIZE, "CPU Seq", i);
            }

            // [TEST 2: GPU Seq]
            CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
            cudaEventRecord(start);
            gpu_sequential_radix_sort(d_test, (int)SIZE);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float seq_ms; cudaEventElapsedTime(&seq_ms, start, stop);
            t_seq += seq_ms;
            if (t == 0) total_errors += verify_sorting(d_test, (int)SIZE, "GPU Seq", i);

            // [TEST 3: GPU Naive]
            CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
            cudaEventRecord(start);
            gpu_parallel_radix_sort(d_test, (int)SIZE);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float n_ms; cudaEventElapsedTime(&n_ms, start, stop);
            t_naive += n_ms;
            if (t == 0) total_errors += verify_sorting(d_test, (int)SIZE, "GPU Naive", i);

            // [TEST 4: GPU P2 (Shared/Scattered)]
            CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
            cudaEventRecord(start);
            gpu_parallel_radix_sort_opt_p2(d_test, (int)SIZE);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float p2_ms; cudaEventElapsedTime(&p2_ms, start, stop);
            t_p2 += p2_ms;
            if (t == 0) total_errors += verify_sorting(d_test, (int)SIZE, "GPU P2 Opt", i);

            // [TEST 5: GPU P3 (Coalesced)]
            CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
            cudaEventRecord(start);
            gpu_parallel_radix_sort_opt_p3(d_test, (int)SIZE);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float p3_ms; cudaEventElapsedTime(&p3_ms, start, stop);
            t_p3 += p3_ms;
            if (t == 0) total_errors += verify_sorting(d_test, (int)SIZE, "GPU P3 Opt", i);

            // [TEST 6: GPU Thrust]
            CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
            cudaEventRecord(start);
            gpu_library_radix_sort(d_test, (int)SIZE);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float lib_ms; cudaEventElapsedTime(&lib_ms, start, stop);
            t_lib += lib_ms;
            if (t == 0) total_errors += verify_sorting(d_test, (int)SIZE, "GPU Thrust", i);

            // Cleanup trial
            cudaEventDestroy(start); cudaEventDestroy(stop);
            cudaFree(d_original); cudaFree(d_test);
            delete[] h_original; delete[] h_test;
        }

        // Calculate Averages
        float a_cpu = t_cpu / NUM_TRIALS;
        float a_seq = t_seq / NUM_TRIALS;
        float a_naive = t_naive / NUM_TRIALS;
        float a_p2 = t_p2 / NUM_TRIALS;
        float a_p3 = t_p3 / NUM_TRIALS;
        float a_lib = t_lib / NUM_TRIALS;

        // Logging
        printf("     - P2 Opt: %8.3f ms | P3 Opt: %8.3f ms\n\n", a_p2, a_p3);

        if (csvFile.is_open()) {
            csvFile << SIZE << "," << a_cpu << "," << a_seq << "," << a_naive << "," << a_p2 << "," << a_p3 << "," << a_lib << "\n";
            csvFile.flush(); 
        }
    }

    printf("\n==========================================\n");
    if (total_errors == 0) printf("  FINAL STATUS: ALL CLEAR ✓\n");
    else printf("  FINAL STATUS: %d TOTAL ERRORS FOUND ✗\n", total_errors);
    printf("==========================================\n");

    csvFile.close();
    return 0;
}