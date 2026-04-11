#!/bin/bash
#PBS -N Radix_Sort_Eval
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:30:00
#PBS -P 52001004
#PBS -o RADIX_results.log
#PBS -j oe

# Change to the directory where you submitted the job
cd "$PBS_O_WORKDIR" || exit $?

cd radix || exit $?

echo "--- Loading Environment Modules ---"
module purge

# Load the GCC module
module load gcc/11.2.0

# Load the CUDA module
module load cuda/12.2.2

echo "--- Compiling Project ---"
make clean
make

echo "--- Running Evaluation ---"
# Run the executable from inside the newly created output folder
./output/radix_sort_eval