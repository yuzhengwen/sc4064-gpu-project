#include "utils.h"
#include <vector>

void cpu_sequential_radix_sort(int* h_arr, int n) {
    int max_val = find_max_host(h_arr, n);
    std::vector<int> temp(n);

    // LSD radix sort: process the ones digit, then tens, hundreds, and so on.
    for (int exp = 1; max_val / exp > 0; exp *= 10) {
        int count[10] = {0};

        // Count how many values fall into each digit bucket for this pass.
        for (int i = 0; i < n; i++) {
            count[(h_arr[i] / exp) % 10]++;
        }

        // Convert counts into end positions for a stable placement pass.
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }

        // Walk backward so equal digits keep their original relative order.
        for (int i = n - 1; i >= 0; i--) {
            int digit = (h_arr[i] / exp) % 10;
            temp[count[digit] - 1] = h_arr[i];
            count[digit]--;
        }

        // Copy the stable output back for the next digit pass.
        for (int i = 0; i < n; i++) {
            h_arr[i] = temp[i];
        }
    }
}