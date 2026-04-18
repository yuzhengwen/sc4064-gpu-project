#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <random>

// --- Helper: Random Data Generation ---
void generateRandomData(std::vector<int> &arr, size_t n, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 1000000000);
    arr.clear();
    arr.resize(n);
    for (size_t i = 0; i < n; i++) arr[i] = dist(gen);
}

void runThrustBenchmark(int k) {
    size_t N = 1ULL << k;
    
    // Generate data on Host
    std::vector<int> host_data;
    generateRandomData(host_data, N);

    // Timing with CUDA Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Thrust automatically handles memory allocation and transfer
    // when you initialize a device_vector from a host_vector/std::vector
    thrust::device_vector<int> d_vec = host_data;

    cudaEventRecord(start);

    // The core Library Call
    thrust::sort(d_vec.begin(), d_vec.end());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Metrics Calculation
    double throughput = N / (ms / 1000.0);
    double totalGB = (2.0 * N * sizeof(int) * k) / 1e9;
    double bandwidth = totalGB / (ms / 1000.0);

    std::cout << "2^" << std::left << std::setw(2) << k
              << " | " << std::setw(10) << N
              << " | " << std::setw(12) << ms << " ms"
              << " | " << std::scientific << std::setprecision(2) << std::setw(12) << throughput << " elem/s"
              << " | " << std::fixed << std::setprecision(3) << bandwidth << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "GPU Library Benchmark: thrust::sort" << std::endl;
    std::cout << "k    | N          | GPU Time      | Throughput   | Bandwidth" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int k = 10; k <= 25; ++k) {
        runThrustBenchmark(k);
    }
    return 0;
}