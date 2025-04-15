#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH -J task1
#SBATCH -o task1.out -e task1.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

#compile task1.cu
module load nvidia/cuda/11.8.0
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1


echo "n time_ms last_val" > timing1_1024.txt

for i in {5..14}; do

n=$((2**i))

echo "Running task1 with n=$n"

output=$(./task1 $n 1024)

# Parse each line from output
last_val=$( echo "$output" | sed -n '1p')
time_ms=$( echo "$output" | sed -n '2p')


echo "$i $time_ms $last_val" >> timing1_1024.txt

done

echo "n time_ms last_val" > timing1_512.txt

for i in {5..14}; do

n=$((2**i))

echo "Running task1 with n=$n"

output=$(./task1 $n 512)

# Parse each line from output
last_val=$( echo "$output" | sed -n '1p')
time_ms=$( echo "$output" | sed -n '2p')


echo "$i $time_ms $last_val" >> timing1_512.txt

done

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="
