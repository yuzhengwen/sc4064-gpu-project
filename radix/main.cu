#include <iostream>
#include <vector>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>
#include "utils.h"

// Define the range of sizes to test
const std::vector<int> TEST_SIZES = {100000, 1000000, 10000000};

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

int find_max_host(int* h_arr, int n) {
    int max_val = h_arr[0];
    for (int i = 1; i < n; i++)
        if (h_arr[i] > max_val) max_val = h_arr[i];
    return max_val;
}

void print_sample(const char* label, int* arr, int n, int count, bool is_device) {
    int* h_sample = new int[count];
    if (is_device) {
        CUDA_CHECK(cudaMemcpy(h_sample, arr, count * sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        memcpy(h_sample, arr, count * sizeof(int));
    }
    printf("%s: ", label);
    for (int i = 0; i < count; i++) printf("%d ", h_sample[i]);
    printf("\n");
    delete[] h_sample;
}

__global__ void verify_sorted(int* data, int n, int* is_sorted) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n - 1) {
        if (data[id] > data[id + 1])
            atomicAdd(is_sorted, 1);
    }
}

void verify_and_print(int* d_arr, int n) {
    int* d_is_sorted;
    int h_is_sorted = 0;
    CUDA_CHECK(cudaMalloc(&d_is_sorted, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_is_sorted, &h_is_sorted, sizeof(int), cudaMemcpyHostToDevice));
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    verify_sorted<<<blocksPerGrid, threadsPerBlock>>>(d_arr, n, d_is_sorted);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_is_sorted, d_is_sorted, sizeof(int), cudaMemcpyDeviceToHost));
    printf(h_is_sorted == 0 ? "✓ Sorted correctly\n\n" : "✗ Failed: %d errors\n\n", h_is_sorted);
    cudaFree(d_is_sorted);
}

int main() {
    printf("--- Radix Sort Comprehensive Scaling Evaluation ---\n\n");

    for (int SIZE : TEST_SIZES) {
        printf("==========================================\n");
        printf(" DATASET SIZE: N = %d elements\n", SIZE);
        printf("==========================================\n");

        int* h_original = new int[SIZE];
        int* h_test = new int[SIZE];
        int* d_original;
        int* d_test;
        CUDA_CHECK(cudaMalloc(&d_original, SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_test, SIZE * sizeof(int)));

        generate_random_data(d_original, SIZE);
        CUDA_CHECK(cudaMemcpy(h_original, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToHost));

        // Global Warmup: Ensure hardware is 'hot' before any timed tests
        int* d_warmup;
        CUDA_CHECK(cudaMalloc(&d_warmup, 100 * sizeof(int)));
        gpu_sequential_radix_sort(d_warmup, 100);
        gpu_parallel_radix_sort(d_warmup, 100);
        gpu_library_radix_sort(d_warmup, 100);
        cudaFree(d_warmup);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // [TEST 1: CPU Sequential] - Standard sequential baseline [cite: 20, 23]
        printf("[1] CPU Sequential Sort\n");
        memcpy(h_test, h_original, SIZE * sizeof(int));
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_sequential_radix_sort(h_test, SIZE);
        auto cpu_stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpu_duration = cpu_stop - cpu_start;
        printf("Time: %.3f ms | Throughput: %.2f M-elem/s\n", 
               cpu_duration.count(), (SIZE / (cpu_duration.count() * 1000.0f)));
        // Copy CPU results to the GPU so the verification kernel can read them
        CUDA_CHECK(cudaMemcpy(d_test, h_test, SIZE * sizeof(int), cudaMemcpyHostToDevice));
        verify_and_print(d_test, SIZE);

        // [TEST 2: GPU Sequential] - High-latency sequential on GPU 
        printf("[2] GPU Sequential Sort (<<<1,1>>>)\n");
        CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaEventRecord(start);
        gpu_sequential_radix_sort(d_test, SIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float seq_ms; cudaEventElapsedTime(&seq_ms, start, stop);
        printf("Time: %.3f ms | Throughput: %.2f M-elem/s\n", 
               seq_ms, (SIZE / (seq_ms * 1000.0f)));
        verify_and_print(d_test, SIZE);

        // [TEST 3: GPU Parallel (Naive)] - Manual Parallel Implementation [cite: 25, 26]
        printf("[3] GPU Parallel Sort (Naive Base)\n");
        CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaEventRecord(start);
        gpu_parallel_radix_sort(d_test, SIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float par_ms; cudaEventElapsedTime(&par_ms, start, stop);
        printf("Time: %.3f ms | Throughput: %.2f M-elem/s\n", 
               par_ms, (SIZE / (par_ms * 1000.0f)));
        verify_and_print(d_test, SIZE);

        // [TEST 4: GPU Parallel (Library)] - Professional Reference [cite: 40, 42]
        printf("[4] GPU Parallel Sort (Thrust Library)\n");
        fflush(stdout); // Forces the text to print immediately before the library starts
        CUDA_CHECK(cudaMemcpy(d_test, d_original, SIZE * sizeof(int), cudaMemcpyDeviceToDevice));
        cudaEventRecord(start);
        gpu_library_radix_sort(d_test, SIZE);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float lib_ms; cudaEventElapsedTime(&lib_ms, start, stop);
        printf("Time: %.3f ms | Throughput: %.2f M-elem/s\n", 
               lib_ms, (SIZE / (lib_ms * 1000.0f)));
        verify_and_print(d_test, SIZE);

        // Memory Management
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_original);
        cudaFree(d_test);
        delete[] h_original;
        delete[] h_test;
        printf("\n");
    }
    return 0;
}