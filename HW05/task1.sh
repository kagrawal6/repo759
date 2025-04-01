#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH -J task1
#SBATCH -o task1.out -e task1.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

#compile task1.cu
module load nvidia/cuda/11.8.0
nvcc task1.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1

#run task1
./task1


echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="
