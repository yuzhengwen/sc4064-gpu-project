#include "utils.h"

__global__ void block_inclusive_scan_kernel(int* data, int* block_sums, int n) {
    extern __shared__ int temp[];
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Load one value per thread into shared memory, padding with zero past n.
    temp[threadIdx.x] = (id < n) ? data[id] : 0;
    __syncthreads();

    // Hillis-Steele inclusive scan inside a block: repeatedly add from farther left.
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int val = 0;
        if (threadIdx.x >= offset) val = temp[threadIdx.x - offset];
        __syncthreads();
        if (threadIdx.x >= offset) temp[threadIdx.x] += val;
        __syncthreads();
    }

    if (id < n) data[id] = temp[threadIdx.x];

    // The last thread stores the block total so higher-level passes can offset it.
    if (threadIdx.x == blockDim.x - 1 && block_sums != nullptr) {
        block_sums[blockIdx.x] = temp[threadIdx.x];
    }
}

__global__ void add_block_sums_kernel(int* data, int* block_sums, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (blockIdx.x > 0 && id < n) data[id] += block_sums[blockIdx.x - 1];
}

__global__ void inclusive_to_exclusive_kernel(int* data, int* exclusive_data, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) exclusive_data[id] = (id == 0) ? 0 : data[id - 1];
}

void manual_inclusive_scan(int* d_arr, int n) {
    if (n <= 0) return;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = threadsPerBlock * sizeof(int);

    // One block is the simple case: scan directly in shared memory.
    if (blocksPerGrid <= 1) {
        block_inclusive_scan_kernel<<<1, threadsPerBlock, sharedMemSize>>>(d_arr, nullptr, n);
    } else {
        // Multi-block inclusive scan: scan each block, recursively scan block sums,
        // then add the previous block totals back into every block.
        int* d_block_sums;
        CUDA_CHECK(cudaMalloc(&d_block_sums, blocksPerGrid * sizeof(int)));
        block_inclusive_scan_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_arr, d_block_sums, n);
        manual_inclusive_scan(d_block_sums, blocksPerGrid);
        add_block_sums_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_block_sums, n);
        CUDA_CHECK(cudaFree(d_block_sums));
    }
}

void manual_exclusive_scan(int* d_arr, int n) {
    if (n <= 0) return;
    // Convert inclusive scan output into an exclusive scan in a separate buffer.
    manual_inclusive_scan(d_arr, n);

    int* d_temp_exclusive;
    CUDA_CHECK(cudaMalloc(&d_temp_exclusive, n * sizeof(int)));
    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    inclusive_to_exclusive_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_temp_exclusive, n);

    CUDA_CHECK(cudaMemcpy(d_arr, d_temp_exclusive, n * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(d_temp_exclusive));
}