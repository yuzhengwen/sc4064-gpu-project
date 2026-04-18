#!/bin/bash
#PBS -q normal
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:mem=16G
#PBS -l walltime=00:10:00
#PBS -P 52001004
#PBS -N BitSort_Stage5_Thrust

cd $PBS_O_WORKDIR
module load cuda

nvcc -O3 bitonic_thrust.cu -o bitonic_thrust

./bitonic_thrust