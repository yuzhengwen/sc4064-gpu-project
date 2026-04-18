#!/bin/bash
# Submission Script for Project (Quick Sort Parallel)
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:mem=32G
#PBS -l walltime=00:20:00
#PBS -P 52001004
#PBS -N Thrust

cd $PBS_O_WORKDIR
module load cuda

# Compile with O3 optimization
nvcc -O3 thrust.cu -o thrust

./thrust
