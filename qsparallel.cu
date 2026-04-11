#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

// --- Data Generation Helper ---
void generateRandomData(std::vector<int> &arr, size_t n, int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 1000000000);
    arr.clear();
    arr.resize(n);
    for (size_t i = 0; i < n; i++)
        arr[i] = dist(gen);
}

// --- GPU Kernels ---

// Basic swap on device
__device__ void d_swap(int &a, int &b)
{
    int t = a;
    a = b;
    b = t;
}

// Device partition logic
__device__ int d_partition(int *arr, int low, int high)
{
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++)
    {
        if (arr[j] < pivot)
        {
            i++;
            d_swap(arr[i], arr[j]);
        }
    }
    d_swap(arr[i + 1], arr[high]);
    return (i + 1);
}

// Naive Global Memory Iterative QuickSort Kernel
__global__ void quickSortKernel(int *arr, int low, int high)
{
    // Note: This naive version uses a single block for the top level.
    // In a production "Optimized Parallel" version, we would use
    // Parallel Scan/Partitioning across multiple blocks.
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        int stack[64]; // Internal stack to avoid recursion
        int top = -1;

        stack[++top] = low;
        stack[++top] = high;

        while (top >= 0)
        {
            high = stack[top--];
            low = stack[top--];

            int p = d_partition(arr, low, high);

            if (p - 1 > low)
            {
                stack[++top] = low;
                stack[++top] = p - 1;
            }
            if (p + 1 < high)
            {
                stack[++top] = p + 1;
                stack[++top] = high;
            }
        }
    }
}

// --- Benchmarking Logic ---

void runGPUBenchmark(int k)
{
    size_t N = 1ULL << k;
    size_t bytes = N * sizeof(int);

    std::vector<int> host_data;
    generateRandomData(host_data, N);

    int *d_arr;
    cudaMalloc(&d_arr, bytes);
    cudaMemcpy(d_arr, host_data.data(), bytes, cudaMemcpyHostToDevice);

    // Timing with CUDA Events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launching 1 block/1 thread for this naive iterative version
    // Future optimizations will increase block/thread count
    quickSortKernel<<<1, 1>>>(d_arr, 0, (int)N - 1);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Calculate Metrics
    double throughput = N / (ms / 1000.0);
    // Bandwidth estimate: (Read + Write) * N * 4 bytes * log2(N)
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

int main()
{
    std::cout << "k    | N          | GPU Time      | Throughput   | Bandwidth" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int k = 10; k <= 25; ++k)
    {
        runGPUBenchmark(k);
    }
    return 0;
}