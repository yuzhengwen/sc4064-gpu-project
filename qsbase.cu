#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
#include <string>
#include <cmath>

// --- Data Generation & Sorting Logic ---
void generateRandomData(std::vector<int> &arr, size_t n, int seed = 42)
{
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 1000000000);
    arr.clear();
    arr.resize(n);
    for (size_t i = 0; i < n; i++)
        arr[i] = dist(gen);
}

void swap(int &a, int &b)
{
    int t = a;
    a = b;
    b = t;
}

int partition(std::vector<int> &arr, int low, int high)
{
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++)
    {
        if (arr[j] < pivot)
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return (i + 1);
}

void quickSort(std::vector<int> &arr, int low, int high)
{
    if (low < high)
    {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

void runFullBenchmark()
{
    // Table Header [cite: 43]
    std::cout << std::left << std::setw(4) << "k"
              << " | " << std::setw(10) << "N"
              << " | " << std::setw(12) << "Manual(ms)"
              << " | " << std::setw(12) << "std(ms)"
              << " | " << std::setw(14) << "Man Elem/s"
              << " | " << std::setw(14) << "std Elem/s"
              << " | " << std::setw(10) << "Man GB/s"
              << " | " << std::setw(10) << "std GB/s" << std::endl;
    std::cout << std::string(110, '-') << std::endl;

    for (int k = 10; k <= 25; ++k)
    {
        size_t N = 1ULL << k;
        std::vector<int> host_data;
        generateRandomData(host_data, N);

        // 1. Benchmark Manual Quick Sort [cite: 16, 44]
        std::vector<int> manual_data = host_data;
        double m_time = 0;
        bool run_m = (k <= 18); // Prevent Stack Overflow [cite: 8, 9]
        if (run_m)
        {
            auto s = std::chrono::high_resolution_clock::now();
            quickSort(manual_data, 0, (int)N - 1);
            auto e = std::chrono::high_resolution_clock::now();
            m_time = std::chrono::duration<double, std::milli>(e - s).count();
        }

        // 2. Benchmark std::sort (Introsort) [cite: 24, 40]
        std::vector<int> std_data = host_data;
        auto s_std = std::chrono::high_resolution_clock::now();
        std::sort(std_data.begin(), std_data.end());
        auto e_std = std::chrono::high_resolution_clock::now();
        double s_time = std::chrono::duration<double, std::milli>(e_std - s_std).count();

        // 3. Calculate Throughput (Elements per second)
        double m_tp = run_m ? (N / (m_time / 1000.0)) : 0;
        double s_tp = N / (s_time / 1000.0);

        // 4. Calculate Achieved Bandwidth (GB/s)
        // Estimate: (Read + Write) * N * 4 bytes * log2(N) passes
        double totalBytes = 2.0 * N * sizeof(int) * k;
        double m_bw = run_m ? (totalBytes / 1e9) / (m_time / 1000.0) : 0;
        double s_bw = (totalBytes / 1e9) / (s_time / 1000.0);

        // Output results
        std::cout << "2^" << std::left << std::setw(2) << k
                  << " | " << std::setw(10) << N
                  << " | " << std::setw(12) << (run_m ? std::to_string(m_time) : "Stack Risk")
                  << " | " << std::setw(12) << s_time
                  << " | " << std::scientific << std::setprecision(2) << std::setw(14) << m_tp
                  << " | " << std::setw(14) << s_tp
                  << " | " << std::fixed << std::setprecision(3) << std::setw(10) << m_bw
                  << " | " << std::setw(10) << s_bw << std::endl;
    }
}

int main()
{
    runFullBenchmark();
    return 0;
}