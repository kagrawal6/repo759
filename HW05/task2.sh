#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH -J task2
#SBATCH -o task2.out -e task2.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

#compile task2.cu
module load nvidia/cuda/11.8.0
nvcc task2.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2

#run task1
./task2


echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="
