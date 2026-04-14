#!/bin/bash
#PBS -N Radix_Sort_Eval
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:30:00
#PBS -P 52001004
#PBS -o RADIX_results.log
#PBS -j oe

# 1. Start in the Root folder
cd "$PBS_O_WORKDIR" || exit $?

echo "--- Loading Environment Modules ---"
module purge
module load gcc/11.2.0
module load cuda/12.2.2

echo "--- Compiling Project ---"
# 2. Go into radix and build
cd radix
make clean
make

# 3. Move the entire output folder from 'radix/output' to the 'Root'
# If a root output folder already exists, we remove it first to avoid errors
rm -rf ../output
mv output ../

# 4. Step back out to the Root folder
cd ..

echo "--- Running Evaluation from Root ---"
# 5. Run the executable now located in the Root's output folder
# This ensures phase1_results.csv is created in the Root
./output/radix_sort_eval 

echo "--- Evaluation Finished ---"