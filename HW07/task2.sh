#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH -J task2
#SBATCH -o task2.out -e task2.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

#compile task2.cu
module load nvidia/cuda/11.8.0

nvcc task2.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2

echo "n time_ms " > timing2_512.txt

for i in {10..30}; do

n=$((2**i))
echo "Running task2 with n=$n"
output=$(./task2 $n 512)
time_ms=$( echo "$output" | sed -n '2p')

echo "$i $time_ms" >> timing2_512.txt


done
