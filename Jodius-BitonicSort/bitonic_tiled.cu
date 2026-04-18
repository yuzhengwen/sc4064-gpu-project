#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>
#include <random>

#define SHARED_SIZE 1024

// --- Data Generation Helper ---
void generateRandomData(std::vector<int> &arr, size_t n, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 1000000000);
    arr.clear(); arr.resize(n);
    for (size_t i = 0; i < n; i++) arr[i] = dist(gen);
}

// --- Kernel 1: Shared Memory Tiled (Handles k <= 1024) ---
__global__ void bitonic_tiled_kernel(int *arr) {
    __shared__ int sh_arr[SHARED_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;

    sh_arr[tid] = arr[i];
    __syncthreads();

    for (int k = 2; k <= SHARED_SIZE; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            unsigned int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0 && sh_arr[tid] > sh_arr[ixj]) {
                    int t = sh_arr[tid]; sh_arr[tid] = sh_arr[ixj]; sh_arr[ixj] = t;
                }
                else if ((tid & k) != 0 && sh_arr[tid] < sh_arr[ixj]) {
                    int t = sh_arr[tid]; sh_arr[tid] = sh_arr[ixj]; sh_arr[ixj] = t;
                }
            }
            __syncthreads();
        }
    }
    arr[i] = sh_arr[tid];
}

// --- Kernel 2: Global Memory (Handles k > 1024) ---
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

    // Phase 1: Shared Memory Tiling for chunks up to 1024
    // We only need to run this once! It completely sorts every 1024-element block.
    int num_blocks = (N + SHARED_SIZE - 1) / SHARED_SIZE;
    bitonic_tiled_kernel<<<num_blocks, SHARED_SIZE>>>(d_arr);

    // Phase 2: Global Memory for the remaining massive merges (k = 2048, 4096, etc.)
    if (N > SHARED_SIZE) {
        for (int k = SHARED_SIZE * 2; k <= N; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                // Using 256 threads per block for the global passes is standard
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
    
    // Total mathematical passes remains the same: k*(k+1)/2
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
    std::cout << "Stage 3: Bitonic Sort (Shared Memory Tiled)" << std::endl;
    std::cout << "k  | N          | Time (ms)  | Throughput   | Bandwidth" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    // Scaling up to k=25
    for (int k = 10; k <= 25; ++k) {
        run_benchmark(k);
    }
    return 0;
}