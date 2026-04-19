# Comparative Analysis of Sorting Architectures: QuickSort 

This directory contains the CUDA C++ source files (`.cu`) and their corresponding PBS submission scripts (`.sh`) for the **QuickSort**  benchmarks evaluated on NVIDIA A100 GPUs.

---

## 📂 File Overview

Each algorithm implementation has a `.cu` source file and a matching `.sh` script used to compile and submit the job to the cluster.

### QuickSort Implementations
* **`qsbase.sh`** (`qsbase.cu`): CPU Baselines (standard `std::sort` and manual CPU QuickSort).
* **`qsparallelop2.sh`** (`qsparallelop2.cu`): Iterative Tiled GPU QuickSort.
* **`qsparallelop.sh`** (`qsparallelop.cu`): Optimized GPU QuickSort (Dynamic Parallelism).
* **`thrust.sh`** (`thrust.cu`): NVIDIA Thrust Library Baseline.

---

## How to Run the Benchmarks

All module loading, compilation flags (e.g., `-O3`, `-arch=sm_70`), and resource requests are handled automatically by the `.sh` files. 

**You must run each script individually to generate the output for that specific benchmark.**

To execute a benchmark, submit its corresponding `.sh` file using the `qsub` command:

### Submit QuickSort Jobs:
```bash
qsub qsbase.sh
qsub qsparallelop2.sh
qsub qsparallelop.sh
qsub thrust.sh
```
## 📈 Viewing the Output
Once the jobs finish running on the compute nodes, the PBS scheduler will output the benchmark results into standard output files in your current directory.

Look for files ending in .o followed by the job ID (e.g., Project_QuickSortBase.o1234567). You can view the benchmark metrics (Execution Time, Throughput, and Achieved Bandwidth) by reading these files.

## How to Generate roofline models
For quicksort:
```bash
python rooflinemodel_seq.py
```
