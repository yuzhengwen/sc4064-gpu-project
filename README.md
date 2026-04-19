# SC4064 GPU Sorting Algorithms — Team 6

Comparative performance analysis of four classical sorting algorithms implemented across CPU and GPU using CUDA, run on the **NTU Aspire2A** cluster (NVIDIA A100). Each algorithm is benchmarked at increasing dataset sizes (up to N = 33.5 × 10⁶ elements) to measure execution time, throughput, memory bandwidth utilisation, and speedup over a CPU baseline.

## Objective

Directly porting sequential algorithms to GPU architectures often causes severe performance degradation due to branching and memory overhead. This project systematically evaluates CPU baselines against custom parallel GPU implementations and the NVIDIA Thrust library to understand _why_ — identifying architectural mismatches (thread divergence, parallelism collapse, memory bandwidth saturation) and how different algorithms overcome them.

## Algorithms

| Folder | Algorithm | Key Insight | Best Custom Speedup vs CPU |
| --- | --- | --- | --- |
| [Jodius-BitonicSort](Jodius-BitonicSort/bitonicsort.md) | Bitonic Sort | Branchless sorting network saturates A100 memory bandwidth wall (~1.3 ms at N=2²⁰) | ~1,000,000× (Tiled Shared Mem) |
| [Jodius-QuickSort](Jodius-QuickSort/quicksort.md) | QuickSort | Thread divergence from dynamic pivots serializes warps; GPU _slower_ than CPU without Thrust | 0.18× (Dynamic Parallelism) |
| [Zivan-radix](Zivan-radix/README.md) | Radix Sort | Non-comparative data-parallel design overcomes bottlenecks; shared memory histograms key step | 19.9× (P3 Coalesced) |
| [Zw-Merge](Zw-Merge/README.md) | Merge Sort | Naive GPU collapses to 1 thread at final pass; parallel co-rank (BSearch) resolves bottleneck | 122× (BSearch) |

## Key Findings

**QuickSort** — Branch-heavy divide-and-conquer causes catastrophic warp divergence. The iterative GPU kernel (9,161 ms) is 125× slower than CPU `std::sort` (73 ms) at N=2²⁰. Dynamic parallelism reduces this to 411 ms but still cannot beat the CPU. Thrust completes in 0.42 ms.

**Bitonic Sort** — Entirely branchless, zero divergence. The naive GPU kernel immediately achieves 664 M elem/s. Shared memory tiling and warp shuffles push to ~800 M elem/s, hitting the physical memory bandwidth wall of the A100. Both optimized kernels converge at 1.3–1.4 ms, representing the hardware ceiling for this workload.

**Merge Sort** — Naive GPU experiences parallelism collapse: at N=2²⁵, the final pass reduces to a single thread merging 33M elements sequentially, consuming >70% of runtime. Shared memory tiling eliminates the fast early passes but leaves the bottleneck untouched. The BSearch (parallel co-rank) implementation resolves this, achieving 1,034 M elem/s and a 351× improvement over naive GPU.

**Radix Sort** — Each optimization stage yields measurable gains. The shared memory histogram kernel (P2) is the largest single step: bandwidth triples to 23.9 GB/s and speedup reaches 18.6×. Coalesced writes (P3) add a further ~7% to 25.6 GB/s. Thrust achieves 433.7 GB/s (28% of hardware peak) via 8-bit radix, warp-level prefix sums, and occupancy tuning.

## Performance Summary (N ≈ 33.5M elements)

| Algorithm | Best Custom GPU Time | Thrust Time | Best Custom Speedup vs CPU |
| --- | --- | --- | --- |
| QuickSort | 411 ms (Dynamic) | 0.42 ms | ~0.18× |
| Bitonic Sort | 1.3 ms (Tiled) | 0.43 ms | ~1,000,000× |
| Merge Sort | 32.5 ms (BSearch) | 2.9 ms | 122× |
| Radix Sort | 42 ms (P3 Coalesced) | 2.5 ms | 19.9× |

## Contributors

| Name | Student ID | Contribution |
| --- | --- | --- |
| Low Zhan Qi (Jodius) | U2221134H | Bitonic Sort, QuickSort |
| Yu Zheng Wen | U2322264J | Merge Sort |
| Zivan Soh Hung Beng | U2421341H | Radix Sort |

## Reports

| Document | Description |
| --- | --- |
| [Team6-Report.pdf](Team6-Report.pdf) | Full written report covering methodology, results, and analysis |
| [Team6-PresentationSlides.pdf](Team6-PresentationSlides.pdf) | Presentation slides summarising findings |

## Hardware Target

All benchmarks are submitted to an **NVIDIA A100 SXM4 40GB** node on the **NTU Aspire2A** cluster via the PBS scheduler (`qsub`). Hardware specs relevant to analysis: peak memory bandwidth **1,555 GB/s**, peak INT32 throughput **19,500 GOps/s**. Each algorithm folder contains its own `.sh` job scripts with resource flags, module loads, and compilation commands.
