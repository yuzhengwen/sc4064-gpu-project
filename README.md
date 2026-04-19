# SC4064 GPU Sorting Algorithms — Team 6

Comparative performance analysis of four classical sorting algorithms implemented across CPU and GPU (NVIDIA A100) using CUDA, run on the **NTU Aspire2A** cluster. Each algorithm is benchmarked at increasing dataset sizes to measure execution time, throughput, memory bandwidth utilisation, and speedup over a CPU baseline.

## Objective

The goal is to understand how different sorting algorithms translate to GPU parallelism — where some map naturally onto the SIMT execution model and others require significant restructuring to extract performance. All GPU variants are compared against a CPU sequential baseline and the NVIDIA Thrust library as a reference ceiling.

## Algorithms

| Folder | Algorithm | Details |
|---|---|---|
| [Jodius-BitonicSort](Jodius-BitonicSort/bitonicsort.md) | Bitonic Sort | Naive global memory → shared memory tiled → warp shuffle → Thrust |
| [Jodius-QuickSort](Jodius-QuickSort/quicksort.md) | QuickSort | CPU baseline → iterative tiled GPU → dynamic parallelism → Thrust |
| [Zivan-radix](Zivan-radix/README.md) | Radix Sort | CPU baseline → GPU naive → shared memory (P2) → coalesced (P3) → Thrust |
| [Zw-Merge](Zw-Merge/README.md) | Merge Sort | CPU baseline → GPU naive → shared memory → binary search → Thrust |

## Contributors

| Name | Contribution |
| --- | --- |
| Low Zhan Qi (Jodius) | Bitonic Sort, QuickSort |
| Yu Zheng Wen | Merge Sort |
| Zivan Soh Hung Beng | Radix Sort |

## Reports

| Document | Description |
| --- | --- |
| [Team6-Report.pdf](Team6-Report.pdf) | Full written report covering methodology, results, and analysis |
| [Team6-PresentationSlides.pdf](Team6-PresentationSlides.pdf) | Presentation slides summarising findings |

## Hardware Target

All benchmarks are submitted to an **NVIDIA A100** node on the **NTU Aspire2A** cluster via the PBS scheduler (`qsub`). Each algorithm folder contains its own `.sh` job scripts with the required resource flags, module loads, and compilation commands.
