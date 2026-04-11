#include "utils.h"
#include <vector>

void cpu_sequential_radix_sort(int* h_arr, int n) {
    int max_val = find_max_host(h_arr, n);
    std::vector<int> temp(n);

    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        int count[10] = {0};

        // Count occurrences
        for (int i = 0; i < n; i++) {
            count[(h_arr[i] / exp) % 10]++;
        }

        // Prefix sum
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }

        // Build output array
        for (int i = n - 1; i >= 0; i--) {
            int digit = (h_arr[i] / exp) % 10;
            temp[count[digit] - 1] = h_arr[i];
            count[digit]--;
        }

        // Copy back to original array
        for (int i = 0; i < n; i++) {
            h_arr[i] = temp[i];
        }
    }
}