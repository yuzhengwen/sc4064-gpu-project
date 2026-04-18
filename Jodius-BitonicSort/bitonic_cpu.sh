#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=1:mem=8G
#PBS -l walltime=00:10:00
#PBS -P 52001004
#PBS -N BitSort_Stage1_CPU

cd $PBS_O_WORKDIR
module load cuda
nvcc -O3 bitonic_cpu.cu -o bitonic_cpu
./bitonic_cpu