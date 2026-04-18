#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024 

// --- Helper: Random Data Generation ---
void generateRandomData(std::vector<int> &arr, size_t n, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 1000000000);
    arr.clear();
    arr.resize(n);
    for (size_t i = 0; i < n; i++) arr[i] = dist(gen);
}

// --- Coalesced & Tiled Partition ---
__device__ int partitionTiled(int *data, int low, int high) {
    __shared__ int tile[BLOCK_SIZE];
    int n = high - low + 1;
    int tid = threadIdx.x;

    // Memory Coalescing: Contiguous load
    if (tid < n) tile[tid] = data[low + tid];
    __syncthreads();

    if (tid == 0) {
        int pivot = tile[n - 1];
        int i = -1;
        for (int j = 0; j < n - 1; j++) {
            if (tile[j] < pivot) {
                i++;
                int temp = tile[i]; tile[i] = tile[j]; tile[j] = temp;
            }
        }
        int temp = tile[i + 1]; tile[i + 1] = tile[n - 1]; tile[n - 1] = temp;
        
        // Coalesced write-back
        for (int k = 0; k < n; k++) data[low + k] = tile[k];
        return low + i + 1;
    }
    __syncthreads();
    return 0;
}

// --- Robust Iterative Kernel (Avoids Dynamic Parallelism Crashes) ---
__global__ void quickSortIterativeTiled(int *arr, int low, int high) {
    if (threadIdx.x == 0) {
        int stack[128]; // Increased stack size for safety
        int top = -1;

        stack[++top] = low;
        stack[++top] = high;

        while (top >= 0) {
            int h = stack[top--];
            int l = stack[top--];

            // Use Tiling for small segments, otherwise global partition
            int p;
            if (h - l + 1 <= BLOCK_SIZE) {
                p = partitionTiled(arr, l, h);
            } else {
                // Global fallback for partitions larger than the shared tile
                int pivot = arr[h];
                int i = l - 1;
                for (int j = l; j < h; j++) {
                    if (arr[j] < pivot) {
                        i++;
                        int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
                    }
                }
                int t = arr[i+1]; arr[i+1] = arr[h]; arr[h] = t;
                p = i + 1;
            }

            if (p - 1 > l) {
                stack[++top] = l;
                stack[++top] = p - 1;
            }
            if (p + 1 < h) {
                stack[++top] = p + 1;
                stack[++top] = h;
            }
        }
    }
}

void runBenchmark(int k) {
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

    // Launch with BLOCK_SIZE threads to support the __shared__ tile operations
    quickSortIterativeTiled<<<1, BLOCK_SIZE>>>(d_arr, 0, (int)N - 1);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error for k=" << k << ": " << cudaGetErrorString(err) << std::endl;
    }

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double throughput = (ms > 0) ? (N / (ms / 1000.0)) : 0;
    double totalGB = (2.0 * N * sizeof(int) * k) / 1e9;
    double bandwidth = (ms > 0) ? (totalGB / (ms / 1000.0)) : 0;

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
    std::cout << "Iterative Tiled QuickSort (Stabilized)" << std::endl;
    std::cout << "k    | N          | GPU Time      | Throughput   | Bandwidth" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int k = 10; k <= 20; ++k) {
        runBenchmark(k);
    }
    return 0;
}