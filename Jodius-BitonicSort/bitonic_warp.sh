#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:mem=16G
#PBS -l walltime=00:10:00
#PBS -P 52001004
#PBS -N BitSort_Stage4_Warp

cd $PBS_O_WORKDIR
module load cuda

nvcc -O3 bitonic_warp.cu -o bitonic_warp

./bitonic_warp