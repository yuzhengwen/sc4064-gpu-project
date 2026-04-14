#include "utils.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

void gpu_library_radix_sort(int* d_arr, int n) {
    // Wrap the raw device pointer so Thrust can treat it as a device range.
    thrust::device_ptr<int> dev_ptr(d_arr);
    
    // Thrust performs the radix sort internally and keeps the work on the GPU.
    thrust::sort(thrust::device, dev_ptr, dev_ptr + n);
}