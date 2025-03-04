#!/usr/bin/env zsh

#SBATCH -p instruction
#SBATCH --cpus-per-task=20
#SBATCH -J task2run
#SBATCH -o task2run.out -e task2run.err

echo "=== Slurm job $SLURM_JOBID started on $(hostname) at $(date) ==="

echo "Compiling convolution.cpp and task2.cpp..."
g++ task2.cpp convolution.cpp -Wall -O3 -std=c++17 -o task2 -fopenmp

# 2) Generate timing_data_2.txt
echo "t time_ms first_val last_val" > timing_data_2.txt

for i in {1..20}; do

t=$i

echo "Running task1 with n = 1024 and t = $t .."

output=$(./task1 1024 "$t")

# parse each line from output
time_ms=$( echo "$output" | sed -n '3p' )
first_val=$(echo "$output" | sed -n '1p' )
last_val=$( echo "$output" | sed -n '2p' )

echo "$t $time_ms $first_val $last_val" >> timing_data_2.txt

done

echo "=== Slurm job $SLURM_JOBID finished at $(date) ==="