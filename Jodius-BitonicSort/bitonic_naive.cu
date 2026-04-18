#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
#include <random>

// --- Data Generation Helper ---
void generateRandomData(std::vector<int> &arr, size_t n, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 1000000000);
    arr.clear(); arr.resize(n);
    for (size_t i = 0; i < n; i++) arr[i] = dist(gen);
}

__global__ void bitonic_naive_kernel(int *arr, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    if (ixj > i) {
        if ((i & k) == 0) {
            if (arr[i] > arr[ixj]) {
                int temp = arr[i];
                arr[i] = arr[ixj];
                arr[ixj] = temp;
            }
        } else {
            if (arr[i] < arr[ixj]) {
                int temp = arr[i];
                arr[i] = arr[ixj];
                arr[ixj] = temp;
            }
        }
    }
}

void run_benchmark(int k_val) {
    size_t N = 1ULL << k_val;
    size_t bytes = N * sizeof(int);

    // Use random data instead of all 1s
    std::vector<int> h_arr;
    generateRandomData(h_arr, N);

    int *d_arr;
    cudaMalloc(&d_arr, bytes);
    cudaMemcpy(d_arr, h_arr.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_naive_kernel<<<(N + 255) / 256, 256>>>(d_arr, j, k);
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // --- Calculate Metrics ---
    double throughput = N / (ms / 1000.0);
    
    // Bitonic sort does k*(k+1)/2 passes. Each pass reads and writes N elements (4 bytes each).
    double passes = (k_val * (k_val + 1)) / 2.0;
    double totalGB = (2.0 * bytes * passes) / 1e9; // 2.0 accounts for 1 read + 1 write
    double bandwidth = totalGB / (ms / 1000.0);

    std::cout << std::left << std::setw(2) << k_val
              << " | " << std::setw(10) << N
              << " | " << std::setw(10) << ms << " ms"
              << " | " << std::scientific << std::setprecision(2) << std::setw(12) << throughput << " elem/s"
              << " | " << std::fixed << std::setprecision(3) << bandwidth << " GB/s" << std::endl;

    cudaFree(d_arr);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    std::cout << "k  | N          | Time (ms)  | Throughput   | Bandwidth" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    for (int k = 10; k <= 25; ++k) {
        run_benchmark(k);
    }
    return 0;
}