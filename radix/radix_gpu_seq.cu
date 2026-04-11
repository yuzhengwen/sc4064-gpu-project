#include "utils.h"

__global__ void sequential_radix_kernel(int* data, int* temp, int n, int exp) {
    int count[10] = {0};

    for (int i = 0; i < n; i++)
        count[(data[i] / exp) % 10]++;

    int prefix[10] = {0};
    for (int i = 1; i < 10; i++)
        prefix[i] = prefix[i-1] + count[i-1];

    for (int i = 0; i < n; i++) {
        int digit = (data[i] / exp) % 10;
        temp[prefix[digit]++] = data[i];
    }
}

void gpu_sequential_radix_sort(int* d_arr, int n) {
    int* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(int)));

    // Copy to host to find max efficiently, then copy back
    int* h_arr = new int[n];
    CUDA_CHECK(cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost));
    int max_val = find_max_host(h_arr, n);
    delete[] h_arr;

    int* src = d_arr;
    int* dst = d_temp;
    int swaps = 0;

    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        sequential_radix_kernel<<<1, 1>>>(src, dst, n, exp);
        CUDA_CHECK(cudaDeviceSynchronize());

        int* tmp = src; src = dst; dst = tmp;
        swaps++;
    }

    if (swaps % 2 != 0) {
        CUDA_CHECK(cudaMemcpy(d_arr, src, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }

    cudaFree(d_temp);
}