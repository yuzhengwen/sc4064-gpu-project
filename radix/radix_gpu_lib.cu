#include "utils.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

void gpu_library_radix_sort(int* d_arr, int n) {
    // Wrap the raw device pointer so Thrust knows it lives on the GPU
    thrust::device_ptr<int> dev_ptr(d_arr);
    
    // Execute the sort entirely on the device
    thrust::sort(thrust::device, dev_ptr, dev_ptr + n);
}