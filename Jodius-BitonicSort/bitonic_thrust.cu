#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>

// --- Data Generation Helper ---
void generateRandomData(std::vector<int> &arr, size_t n, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 1000000000);
    arr.clear(); arr.resize(n);
    for (size_t i = 0; i < n; i++) arr[i] = dist(gen);
}

void run_benchmark(int k_val) {
    size_t N = 1ULL << k_val;
    size_t bytes = N * sizeof(int);

    std::vector<int> h_arr;
    generateRandomData(h_arr, N);

    // Thrust automatically handles the memory allocation and copying here
    thrust::device_vector<int> d_vec = h_arr;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Execute the Gold Standard sort
    thrust::sort(d_vec.begin(), d_vec.end());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // --- Calculate Metrics ---
    double throughput = N / (ms / 1000.0);
    
    // NOTE: We apply the Bitonic Sort complexity here for a 1:1 baseline comparison
    double passes = (k_val * (k_val + 1)) / 2.0; 
    double totalGB = (2.0 * bytes * passes) / 1e9; 
    double bandwidth = totalGB / (ms / 1000.0);

    std::cout << std::left << std::setw(2) << k_val
              << " | " << std::setw(10) << N
              << " | " << std::setw(10) << ms << " ms"
              << " | " << std::scientific << std::setprecision(2) << std::setw(12) << throughput << " elem/s"
              << " | " << std::fixed << std::setprecision(3) << bandwidth << " GB/s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "Stage 5: Industry Standard (NVIDIA Thrust)" << std::endl;
    std::cout << "k  | N          | Time (ms)  | Throughput   | Bandwidth" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    // Scaling up to k=25
    for (int k = 10; k <= 25; ++k) {
        run_benchmark(k);
    }
    return 0;
}