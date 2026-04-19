# Zivan-radix

## Overview

This folder contains a radix sort evaluation project with one CPU baseline and multiple GPU implementations, plus a visualization workflow for performance analysis.

The benchmark driver runs all variants across increasing dataset sizes, verifies correctness, and writes CSV results for plotting.

## Implementation Map

- Benchmark entrypoint: [code/main.cu](code/main.cu)
- Shared declarations and helpers: [code/utils.h](code/utils.h)
- Prefix-scan utilities used by GPU paths: [code/scan.cu](code/scan.cu)
- CPU sequential radix baseline: [code/radix_cpu_seq.cpp](code/radix_cpu_seq.cpp)
- GPU sequential baseline (single-thread kernel): [code/radix_gpu_seq.cu](code/radix_gpu_seq.cu)
- GPU parallel baseline (predicate + scan + scatter): [code/radix_gpu_par.cu](code/radix_gpu_par.cu)
- GPU optimized phase 2 (block histogram + reorder): [code/radix_gpu_opt_p2.cu](code/radix_gpu_opt_p2.cu)
- GPU optimized phase 3 (coalesced tile reorder): [code/radix_gpu_opt_p3.cu](code/radix_gpu_opt_p3.cu)
- GPU library reference (Thrust sort): [code/radix_gpu_lib.cu](code/radix_gpu_lib.cu)
- Build rules and object/executable targets: [Makefile](Makefile)
- Cluster batch script for compile + run: [job_radix.sh](job_radix.sh)

## Project Structure

- Source files: [code/](code/)
- Build output directory: [output/](output/)
- Visualization assets and notebook: [visualization/](visualization/)
  - Notebook: [visualization/visualization.ipynb](visualization/visualization.ipynb)
  - CSV results: [visualization/phase2_averaged_results.csv](visualization/phase2_averaged_results.csv), [visualization/phase3_averaged_results.csv](visualization/phase3_averaged_results.csv), [visualization/phase4_results.csv](visualization/phase4_results.csv)
  - Example figure: [visualization/phase4_comparison_plots.png](visualization/phase4_comparison_plots.png)

## How To Run (PBS via job_radix.sh)

The intended workflow is to submit the batch script, which does all of the following:

1. Loads required modules (GCC and CUDA).
2. Builds the project with Makefile.
3. Runs the benchmark executable.
4. Moves generated phase CSV files into the visualization folder.

From inside this folder:

```bash
qsub job_radix.sh
```

After job completion:

- Log file is written to [output/RADIX_results.log](output/RADIX_results.log)
- Executable is at [output/radix_sort_eval](output/radix_sort_eval)
- CSV outputs are placed in [visualization/](visualization/)

## Optional Local Run (without PBS)

If you are on a machine with NVCC/CUDA configured:

```bash
make clean
make
./output/radix_sort_eval
mv phase*.csv visualization/
```

## Visualization Workflow

Use the notebook in the visualization folder: [visualization/visualization.ipynb](visualization/visualization.ipynb)

Recommended steps:

1. Open the notebook.
2. Ensure Python packages for plotting are available (pandas, matplotlib, numpy).
3. Run cells from top to bottom.
4. Confirm the notebook is reading the expected CSV files in [visualization/](visualization/).

The notebook includes plotting for:

- Execution time scaling
- Throughput scalability
- Speedup vs CPU baseline
- Bandwidth utilization
- Roofline-style analysis

## Notes

- The Makefile target executable is [output/radix_sort_eval](output/radix_sort_eval).
- The batch script creates output and visualization directories if needed.
- Existing CSV files in visualization can be replaced by new runs.
