#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

#define THRESHOLD 32

// Helper for random data generation
void generateRandomData(std::vector<int> &arr, size_t n, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 1000000000);
    arr.clear();
    arr.resize(n);
    for (size_t i = 0; i < n; i++) arr[i] = dist(gen);
}

__device__ void d_swap(int &a, int &b) {
    int t = a; a = b; b = t;
}

// Optimized Selection/Insertion sort for small partitions to reduce kernel overhead
__device__ void smallSort(int *arr, int low, int high) {
    for (int i = low + 1; i <= high; i++) {
        int val = arr[i];
        int j = i - 1;
        while (j >= low && arr[j] > val) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = val;
    }
}

__device__ int d_partition(int *arr, int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            d_swap(arr[i], arr[j]);
        }
    }
    d_swap(arr[i + 1], arr[high]);
    return (i + 1);
}

// Optimized Kernel using Dynamic Parallelism
__global__ void quickSortKernel_Opt(int *arr, int low, int high, int depth) {
    if (low >= high) return;

    // Optimization: For small arrays, don't recurse; use a simple sort
    if (high - low < THRESHOLD) {
        smallSort(arr, low, high);
        return;
    }

    int p = d_partition(arr, low, high);

    // Launch child kernels for the two halves (Dynamic Parallelism)
    // We limit depth to prevent excessive overhead on very large arrays
    if (depth < 24) {
        cudaStream_t s1, s2;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        quickSortKernel_Opt<<<1, 1, 0, s1>>>(arr, low, p - 1, depth + 1);
        
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        quickSortKernel_Opt<<<1, 1, 0, s2>>>(arr, p + 1, high, depth + 1);
        
        cudaStreamDestroy(s1);
        cudaStreamDestroy(s2);
    } else {
        // Fallback to sequential device logic if too deep
        quickSortKernel_Opt<<<1, 1>>>(arr, low, p - 1, depth + 1);
        quickSortKernel_Opt<<<1, 1>>>(arr, p + 1, high, depth + 1);
    }
}

void runOptimizedGPUBenchmark(int k) {
    size_t N = 1ULL << k;
    size_t bytes = N * sizeof(int);

    std::vector<int> host_data;
    generateRandomData(host_data, N);

    int *d_arr;
    cudaMalloc(&d_arr, bytes);
    cudaMemcpy(d_arr, host_data.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch top-level kernel
    quickSortKernel_Opt<<<1, 1>>>(d_arr, 0, (int)N - 1, 0);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Standard metrics
    double throughput = N / (ms / 1000.0);
    double totalGB = (2.0 * N * sizeof(int) * k) / 1e9;
    double bandwidth = totalGB / (ms / 1000.0);

    std::cout << "2^" << std::left << std::setw(2) << k
              << " | " << std::setw(10) << N
              << " | " << std::setw(12) << ms << " ms"
              << " | " << std::scientific << std::setprecision(2) << std::setw(12) << throughput << " elem/s"
              << " | " << std::fixed << std::setprecision(3) << bandwidth << " GB/s" << std::endl;

    cudaFree(d_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "Optimized Parallel QuickSort (Dynamic Parallelism + Thresholding)" << std::endl;
    std::cout << "k    | N          | GPU Time      | Throughput   | Bandwidth" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int k = 10; k <= 20; ++k) { // Note: reduced k max slightly for dynamic parallelism overhead
        runOptimizedGPUBenchmark(k);
    }
    return 0;
}