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

// --- Kernel 1: Warp-Level Sort (Handles k <= 32) ---
__device__ void warp_sort(int &val) {
    for (int k = 2; k <= 32; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int target = __shfl_xor_sync(0xffffffff, val, j);
            bool ascending = (threadIdx.x & k) == 0;
            if (ascending && val > target) val = target;
            if (!ascending && val < target) val = target;
        }
    }
}

__global__ void bitonic_warp_kernel(int *arr) {
    int val = arr[threadIdx.x + blockIdx.x * blockDim.x];
    warp_sort(val);
    arr[threadIdx.x + blockIdx.x * blockDim.x] = val;
}

// --- Kernel 2: Global Memory (Handles k > 32) ---
__global__ void bitonic_global_kernel(int *arr, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;
    if (ixj > i) {
        if ((i & k) == 0) {
            if (arr[i] > arr[ixj]) {
                int temp = arr[i]; arr[i] = arr[ixj]; arr[ixj] = temp;
            }
        } else {
            if (arr[i] < arr[ixj]) {
                int temp = arr[i]; arr[i] = arr[ixj]; arr[ixj] = temp;
            }
        }
    }
}

// --- Benchmarking Logic ---
void run_benchmark(int k_val) {
    size_t N = 1ULL << k_val;
    size_t bytes = N * sizeof(int);

    std::vector<int> h_arr;
    generateRandomData(h_arr, N);

    int *d_arr;
    cudaMalloc(&d_arr, bytes);
    cudaMemcpy(d_arr, h_arr.data(), bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Phase 1: Warp Primitives for chunks up to 32
    // We launch with 256 threads per block, which equals 8 independent warps per block
    bitonic_warp_kernel<<<(N + 255) / 256, 256>>>(d_arr);

    // Phase 2: Global Memory for the remaining merges (k = 64, 128... N)
    if (N > 32) {
        for (int k = 64; k <= N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                bitonic_global_kernel<<<(N + 255) / 256, 256>>>(d_arr, j, k);
            }
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // --- Calculate Metrics ---
    double throughput = N / (ms / 1000.0);
    double passes = (k_val * (k_val + 1)) / 2.0; 
    double totalGB = (2.0 * bytes * passes) / 1e9; 
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
    std::cout << "Stage 4: Bitonic Sort (Warp Shuffle Primitives)" << std::endl;
    std::cout << "k  | N          | Time (ms)  | Throughput   | Bandwidth" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    // Scaling up to k=25
    for (int k = 10; k <= 25; ++k) {
        run_benchmark(k);
    }
    return 0;
}