#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH -J task3_16
#SBATCH -o task3_16.out -e task3_16.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

#compile task3.cu
module load nvidia/cuda/11.8.0
nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

echo "i time_ms first_val last_val" > timing_16.txt

for i in {10..29}; do

n=$((2**i))

echo "Running task3 with n=$n"

output=$(./task3 $n)

# Parse each line from output
first_val=$(echo "$output" | sed -n '2p')
last_val=$( echo "$output" | sed -n '3p')
time_ms=$( echo "$output" | sed -n '1p')


echo "$i $time_ms $first_val $last_val" >> timing_16.txt

done


echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="
