#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:mem=16G
#PBS -l walltime=00:10:00
#PBS -P 52001004
#PBS -N BitSort_Stage2_Naive

cd $PBS_O_WORKDIR
module load cuda

nvcc -O3 -arch=sm_70 bitonic_naive.cu -o bitonic_naive

./bitonic_naive