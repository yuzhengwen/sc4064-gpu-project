#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>

void generateRandomData(std::vector<int> &arr, size_t n, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 1000000000);
    arr.clear(); arr.resize(n);
    for (size_t i = 0; i < n; i++) arr[i] = dist(gen);
}

void bitonicSortCPU(std::vector<int>& arr, int low, int cnt, bool dir) {
    if (cnt > 1) {
        int k = cnt / 2;
        bitonicSortCPU(arr, low, k, true);
        bitonicSortCPU(arr, low + k, k, false);
        // Bitonic Merge
        for (int i = low; i < low + k; i++) {
            if (dir == (arr[i] > arr[i + k])) std::swap(arr[i], arr[i + k]);
        }
        bitonicSortCPU(arr, low, k, dir);
        bitonicSortCPU(arr, low + k, k, dir);
    }
}

int main() {
    std::cout << "k  | N | Time (ms) | Throughput (Elem/s) | Bandwidth (GB/s)" << std::endl;
    for (int k = 20; k <= 25; ++k) {
        size_t N = 1ULL << k;
        std::vector<int> data; generateRandomData(data, N);
        auto s = std::chrono::high_resolution_clock::now();
        bitonicSortCPU(data, 0, N, true);
        auto e = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(e - s).count();
        double tp = N / (ms / 1000.0);
        double totalBytes = 2.0 * N * sizeof(int) * (k * (k + 1) / 2.0);
        double bw = (totalBytes / 1e9) / (ms / 1000.0);
        std::cout << k << " | " << N << " | " << ms << " | " << tp << " | " << bw << std::endl;
    }
    return 0;
}