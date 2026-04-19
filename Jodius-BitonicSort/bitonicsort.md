# Comparative Analysis of Sorting Architectures: Bitonic Sort

This directory contains the CUDA C++ source files (`.cu`) and their corresponding PBS submission scripts (`.sh`) for **Bitonic Sort** benchmarks evaluated on NVIDIA A100 GPUs.

---

## 📂 File Overview

Each algorithm implementation has a `.cu` source file and a matching `.sh` script used to compile and submit the job to the cluster.

### Bitonic Sort Implementations
* **`bitonic_cpu.sh`** (`bitonic_cpu.cu`): CPU Baseline for small array sizes ($N \le 2^{19}$).
* **`bitonic_cpu_2.sh`** (`bitonic_cpu_2.cu`): CPU Baseline for large array sizes ($N \ge 2^{20}$).
* **`bitonic_naive.sh`** (`bitonic_naive.cu`): Naive Global Memory GPU implementation.
* **`bitonic_tiled.sh`** (`bitonic_tiled.cu`): Optimized Shared Memory Tiled GPU implementation.
* **`bitonic_warp.sh`** (`bitonic_warp.cu`): Highly optimized Warp Shuffle GPU implementation.
* **`bitonic_thrust.sh`** (`bitonic_thrust.cu`): NVIDIA Thrust Library Baseline.

---

## How to Run the Benchmarks

All module loading, compilation flags (e.g., `-O3`, `-arch=sm_70`), and resource requests are handled automatically by the `.sh` files. 

**You must run each script individually to generate the output for that specific benchmark.**

To execute a benchmark, submit its corresponding `.sh` file using the `qsub` command:

### Submit Bitonic Sort Jobs:
```bash
qsub bitonic_cpu.sh
qsub bitonic_cpu2.sh
qsub bitonic_naive.sh
qsub bitonic_tiled.sh
qsub bitonic_warp.sh
qsub bitonic_thrust.sh
```
## 📈 Viewing the Output
Once the jobs finish running on the compute nodes, the PBS scheduler will output the benchmark results into standard output files in your current directory.

Look for files ending in .o followed by the job ID (e.g., BitSort_Stage3_Tiled.o7654321). You can view the benchmark metrics (Execution Time, Throughput, and Achieved Bandwidth) by reading these files.

## How to Generate roofline models
For bitonic sort:
```bash
python roofline.py
```