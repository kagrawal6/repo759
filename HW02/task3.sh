#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH -t 0-00:10:00
#SBATCH -J task3run
#SBATCH -o task3run.out -e task3run.err

module load gcc

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

g++ task3.cpp matmul.cpp -Wall -O3 -std=c++17 -o task3

echo "Running: ./task3"
./task3

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="
