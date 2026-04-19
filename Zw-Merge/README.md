To run the merge sort variations and benchmark results
Simply run
`qsub job.sh`

Alternatively, manually:
`nvcc -O3 -std=c++14 main.cu input_gen.cu cpu_sort.cu gpu_naive.cu gpu_smem.cu gpu_bsearch.cu gpu_thrust.cu -o merge_sort_benchmark`
`./merge_sort_benchmark`