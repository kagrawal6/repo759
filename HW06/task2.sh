#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH -J task2
#SBATCH -o task2.out -e task2.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

#compile task2.cu
module load nvidia/cuda/11.8.0
nvcc task2.cu stencil.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task2



echo "n time_ms last_val" > timing2_1024.txt

for i in {10..29}; do

n=$((2**i))

echo "Running task2 with n=$n"

output=$(./task2 $n 128 1024)

# Parse each line from output
last_val=$( echo "$output" | sed -n '1p')
time_ms=$( echo "$output" | sed -n '2p')


echo "$i $time_ms $last_val" >> timing2_1024.txt

done

echo "n time_ms last_val" > timing2_512.txt

for i in {10..29}; do

n=$((2**i))

echo "Running task2 with n=$n"

output=$(./task2 $n 128 512)

# Parse each line from output
last_val=$( echo "$output" | sed -n '1p')
time_ms=$( echo "$output" | sed -n '2p')


echo "$i $time_ms $last_val" >> timing2_512.txt

done

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="