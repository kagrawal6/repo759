#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J task2run
#SBATCH -o task2run.out -e task2run.err

# Usage: sbatch task2.sh <n> <m>
# Example: sbatch task2.sh 1024 3

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

# 1) Compile
g++ convolution.cpp task2.cpp -Wall -O3 -std=c++17 -o task2

# 2) Read n, m from command-line arguments
./task2 1024 3


echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="


