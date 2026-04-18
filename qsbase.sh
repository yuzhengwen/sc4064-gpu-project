#!/bin/bash
# Submission Script for Project (Quick Sort Base)
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:mem=32G
#PBS -l walltime=00:20:00
#PBS -P 52001004
#PBS -N Project_QuickSortBase

cd $PBS_O_WORKDIR
module load cuda

# Compile with O3 optimization
nvcc -O3 qsbase.cu -o qsbase

./qsbase
