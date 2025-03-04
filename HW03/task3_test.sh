#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --cpus-per-task=20
#SBATCH -J task3_test
#SBATCH -o task3_test.out -e task3_test.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

echo "Compiling msort.cpp and task3.cpp..."
g++ task3.cpp msort.cpp -Wall -O3 -std=c++17 -o task3 -fopenmp

# Suppose n=10^6, 
# and use the best threshold from your first script
n=100
ts=50  
t=20
./task3 $n $t $ts

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="
