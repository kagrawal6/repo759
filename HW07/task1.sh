#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -J task1
#SBATCH -o task1.out -e task1.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="
#compile task1.cu
module load nvidia/cuda/11.8.0
nvcc task1.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o task1

echo "n time_ms1 time_ms2 time_ms3" > timing1.txt
for i in {5..14}; do
n=$((2**i))

echo "Running task1 with n=$n"

output=$(./task1 $n 16)

time_ms1=$( echo "$output" | sed -n '3p')
time_ms2=$( echo "$output" | sed -n '6p')
time_ms3=$( echo "$output" | sed -n '9p')

echo "$i $time_ms1 $time_ms2 $time_ms3" >> timing1.txt

done
