#!/bin/bash
#PBS -N Radix_Sort_Eval
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=00:30:00
#PBS -P 52001004
#PBS -o output/RADIX_results.log
#PBS -j oe

# In PBS, $0 points to a spool copy, so prefer PBS_O_WORKDIR.
START_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "$0")" && pwd)}"

# Support launching qsub either from project root or from project/code.
if [ -f "$START_DIR/Makefile" ]; then
	PROJECT_DIR="$START_DIR"
elif [ -f "$START_DIR/../Makefile" ]; then
	PROJECT_DIR=$(cd "$START_DIR/.." && pwd)
else
	echo "ERROR: Could not locate project Makefile from START_DIR=$START_DIR"
	exit 1
fi

cd "$PROJECT_DIR" || exit $?

RESULTS_DIR="visualization"

echo "--- Loading Environment Modules ---"
module purge
module load gcc/11.2.0
module load cuda/12.2.2

echo "--- Compiling Project ---"
mkdir -p output "$RESULTS_DIR"
make clean
make

echo "--- Running Evaluation ---"
# Run from PROJECT_DIR so generated CSV files land in this project.
./output/radix_sort_eval

echo "--- Moving CSV outputs to $RESULTS_DIR/ ---"
mv -f phase*.csv "$RESULTS_DIR"/ 2>/dev/null || true

echo "--- Evaluation Finished ---"